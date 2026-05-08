"""
Step 01: Ingest metadata from UNAM repository, OpenAlex, SciELO.
Outputs a unified parquet in data/raw/records.parquet
"""
import argparse
import re
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from semantic_research_atlas.utils import load_config, ensure_dirs

NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
}


def _first_text(parent: ET.Element, tag: str) -> str:
    el = parent.find(tag, NS)
    return el.text.strip() if el is not None and el.text else ""


def _all_text(parent: ET.Element, tag: str) -> List[str]:
    return [el.text.strip() for el in parent.findall(tag, NS) if el.text]


def _parse_year(date_text: str) -> Optional[int]:
    if not date_text:
        return None
    match = re.search(r"(19|20)\d{2}", date_text)
    return int(match.group(0)) if match else None


def _parse_oai_batch(xml_text: str, source_name: str) -> Tuple[List[Dict], Optional[str]]:
    root = ET.fromstring(xml_text)
    records = []

    for record in root.findall(".//oai:record", NS):
        header = record.find("oai:header", NS)
        if header is None:
            continue
        identifier = header.findtext("oai:identifier", default="", namespaces=NS)

        metadata = record.find("oai:metadata", NS)
        if metadata is None:
            continue
        dc = metadata.find("dc:dc", NS)
        if dc is None:
            continue

        title = _first_text(dc, "dc:title")
        abstract = " ".join(_all_text(dc, "dc:description"))
        date_text = _first_text(dc, "dc:date")
        year = _parse_year(date_text)
        identifiers = _all_text(dc, "dc:identifier")
        url = next((i for i in identifiers if i.startswith("http")), "")
        publisher = _first_text(dc, "dc:publisher")
        subject = _first_text(dc, "dc:subject")
        faculty = publisher or subject or ""

        records.append(
            {
                "id": identifier,
                "title": title,
                "year": year,
                "faculty": faculty,
                "abstract": abstract,
                "source": source_name,
                "url": url,
            }
        )

    token_el = root.find(".//oai:resumptionToken", NS)
    token = token_el.text.strip() if token_el is not None and token_el.text else None
    return records, token


def harvest_oai_pmh(
    base_url: str,
    source_name: str,
    metadata_prefix: str = "oai_dc",
    max_records: int = 0,
    pause_seconds: float = 0.2,
) -> pd.DataFrame:
    records: List[Dict] = []
    token: Optional[str] = None

    while True:
        if token:
            params = {"verb": "ListRecords", "resumptionToken": token}
        else:
            params = {"verb": "ListRecords", "metadataPrefix": metadata_prefix}

        response = requests.get(base_url, params=params, timeout=60)
        response.raise_for_status()

        batch, token = _parse_oai_batch(response.text, source_name)
        records.extend(batch)

        if max_records and len(records) >= max_records:
            records = records[:max_records]
            break

        if not token:
            break

        time.sleep(pause_seconds)

    return pd.DataFrame(records)


def _abstract_from_inverted_index(inv: Optional[Dict]) -> str:
    if not inv:
        return ""
    max_pos = max(pos for positions in inv.values() for pos in positions)
    words = [""] * (max_pos + 1)
    for word, positions in inv.items():
        for pos in positions:
            if pos < len(words):
                words[pos] = word
    return " ".join([w for w in words if w])


def ingest_openalex(
    base_url: str,
    ror_id: str,
    per_page: int = 200,
    max_records: int = 0,
) -> pd.DataFrame:
    records: List[Dict] = []
    cursor = "*"

    while True:
        params = {
            "filter": f"institutions.ror:{ror_id}",
            "per-page": per_page,
            "cursor": cursor,
        }
        response = requests.get(f"{base_url}/works", params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()

        for work in payload.get("results", []):
            abstract = _abstract_from_inverted_index(work.get("abstract_inverted_index"))
            landing = (
                work.get("primary_location", {})
                .get("landing_page_url", "")
            )
            records.append(
                {
                    "id": work.get("id", ""),
                    "title": work.get("display_name", ""),
                    "year": work.get("publication_year"),
                    "faculty": "",
                    "abstract": abstract,
                    "source": "openalex",
                    "url": landing or work.get("id", ""),
                }
            )

            if max_records and len(records) >= max_records:
                return pd.DataFrame(records)

        cursor = payload.get("meta", {}).get("next_cursor")
        if not cursor:
            break

    return pd.DataFrame(records)


def main(config_path: str):
    cfg = load_config(config_path)
    ensure_dirs(cfg["paths"]["raw"], cfg["paths"]["processed"], cfg["paths"]["index"])

    max_records = cfg.get("ingest", {}).get("max_records_per_source", 0)
    pause_seconds = cfg.get("ingest", {}).get("polite_pause_seconds", 0.2)

    frames = []

    if cfg["sources"]["unam_repository"]["enabled"]:
        frames.append(
            harvest_oai_pmh(
                base_url=cfg["sources"]["unam_repository"]["base_url"],
                source_name="unam_repository",
                metadata_prefix=cfg["sources"]["unam_repository"]["metadata_prefix"],
                max_records=max_records,
                pause_seconds=pause_seconds,
            )
        )

    if cfg["sources"]["scielo"]["enabled"]:
        frames.append(
            harvest_oai_pmh(
                base_url=cfg["sources"]["scielo"]["base_url"],
                source_name="scielo_mexico",
                metadata_prefix=cfg["sources"]["scielo"]["metadata_prefix"],
                max_records=max_records,
                pause_seconds=pause_seconds,
            )
        )

    if cfg["sources"]["openalex"]["enabled"]:
        frames.append(
            ingest_openalex(
                base_url=cfg["sources"]["openalex"]["base_url"],
                ror_id=cfg["sources"]["openalex"]["ror_id"],
                per_page=cfg["sources"]["openalex"]["per_page"],
                max_records=max_records,
            )
        )

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["id", "title", "year", "faculty", "abstract", "source", "url"]
    )

    df = df.drop_duplicates(subset=["id", "url"]).reset_index(drop=True)

    out_path = f"{cfg['paths']['raw']}/records.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved: {out_path} ({len(df):,} records)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
