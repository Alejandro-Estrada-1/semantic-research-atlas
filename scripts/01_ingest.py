"""
Step 01: Ingest metadata from UNAM repository, OpenAlex, SciELO.
Outputs a unified parquet in data/raw/records.parquet
"""
import argparse
import io
import json
import os
import re
import shutil
import tempfile
import time
import unicodedata
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from semantic_research_atlas.utils import load_config, ensure_dirs

NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
}

OUTPUT_COLUMNS = ["id", "title", "year", "faculty", "abstract", "source", "url"]
RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
UNAM_OPENALEX_ID = "https://openalex.org/I8961855"
UNAM_OPENALEX_ID_OLD = "https://openalex.org/I70126969"  # legacy, may still appear


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_path(checkpoint_root: str, source_name: str) -> str:
    return os.path.join(checkpoint_root, f"{source_name}_checkpoint.json")


def _load_checkpoint(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_checkpoint(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=True, indent=2)


def _delete_checkpoint(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def _window_key(from_date: Optional[str], until_date: Optional[str]) -> str:
    return f"{from_date or 'none'}__{until_date or 'none'}"


def _get_window_checkpoint(
    checkpoint_path: str,
    source_name: str,
    from_date: Optional[str],
    until_date: Optional[str],
) -> Tuple[Optional[str], int, str, bool]:
    data = _load_checkpoint(checkpoint_path)
    if not data:
        return None, 0, _utc_now(), False

    windows = data.get("windows", {})
    key = _window_key(from_date, until_date)
    window_state = windows.get(key, {})
    if window_state.get("completed"):
        return None, window_state.get("records_collected", 0), _utc_now(), True
    return (
        window_state.get("resumption_token"),
        window_state.get("records_collected", 0),
        window_state.get("harvest_started_at", _utc_now()),
        False,
    )


def _update_window_checkpoint(
    checkpoint_path: str,
    source_name: str,
    from_date: Optional[str],
    until_date: Optional[str],
    resumption_token: Optional[str],
    records_collected: int,
    harvest_started_at: str,
    completed: bool = False,
) -> None:
    data = _load_checkpoint(checkpoint_path) or {"source": source_name, "windows": {}}
    windows = data.setdefault("windows", {})
    key = _window_key(from_date, until_date)
    windows[key] = {
        "source": source_name,
        "from_date": from_date,
        "until_date": until_date,
        "resumption_token": resumption_token,
        "harvest_started_at": harvest_started_at,
        "last_batch_at": _utc_now(),
        "records_collected": records_collected,
        "completed": completed,
    }
    data["last_updated_at"] = _utc_now()
    _save_checkpoint(checkpoint_path, data)


def _all_windows_completed(checkpoint_path: str, windows: List[Tuple[str, str]]) -> bool:
    data = _load_checkpoint(checkpoint_path)
    if not data:
        return True
    window_data = data.get("windows", {})
    for from_date, until_date in windows:
        key = _window_key(from_date, until_date)
        if not window_data.get(key, {}).get("completed"):
            return False
    return True


def _record_from_element(record: ET.Element, source_name: str) -> Optional[Dict]:
    header = record.find("oai:header", NS)
    if header is None or header.get("status") == "deleted":
        return None
    identifier = header.findtext("oai:identifier", default="", namespaces=NS)

    metadata = record.find("oai:metadata", NS)
    if metadata is None:
        return None
    dc = metadata.find("oai_dc:dc", NS)
    if dc is None:
        dc = metadata.find("dc:dc", NS)
    if dc is None:
        return None

    title = _first_text(dc, "dc:title")
    abstract = " ".join(_all_text(dc, "dc:description"))
    date_text = _first_text(dc, "dc:date")
    year = _parse_year(date_text)
    identifiers = _all_text(dc, "dc:identifier")
    url = next((i for i in identifiers if i.startswith("http")), "")
    publisher = _first_text(dc, "dc:publisher")
    subject = _first_text(dc, "dc:subject")
    faculty = publisher or subject or ""

    return {
        "id": identifier,
        "title": title,
        "year": year,
        "faculty": faculty,
        "abstract": abstract,
        "source": source_name,
        "url": url,
    }


def _parse_oai_streaming(
    xml_content: bytes,
    source_name: str,
) -> Tuple[List[Dict], Optional[str]]:
    records: List[Dict] = []
    token = None

    try:
        context = ET.iterparse(io.BytesIO(xml_content), events=("end",))
        for _, elem in context:
            if elem.tag.endswith("record"):
                record = _record_from_element(elem, source_name)
                if record:
                    records.append(record)
                elem.clear()
            elif elem.tag.endswith("resumptionToken"):
                if elem.text:
                    token = elem.text.strip()
                elem.clear()
    except ET.ParseError as exc:
        snippet = xml_content[:500].decode("utf-8", errors="replace")
        print(
            f"[{source_name}] XML ParseError: {exc}. "
            f"Response snippet (first 500 chars): {snippet}"
        )
        raise

    return records, token


def _http_get_with_retries(
    session: requests.Session,
    url: str,
    params: Dict,
    source_name: str,
    request_label: str,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> requests.Response:
    attempt = 0
    headers = {
        "User-Agent": "semantic-research-atlas/0.1 (research metadata harvester)",
        "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
    }

    while True:
        try:
            response = session.get(
                url,
                params=params,
                timeout=timeout_seconds,
                headers=headers,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code is not None and status_code not in RETRYABLE_STATUS_CODES:
                raise

            if attempt >= max_retries:
                raise

            attempt += 1
            response = getattr(exc, "response", None)
            retry_after = None
            if response is not None and response.status_code == 503:
                retry_after = response.headers.get("Retry-After")

            if retry_after:
                try:
                    wait_seconds = float(retry_after)
                except ValueError:
                    wait_seconds = retry_backoff_seconds * (2 ** (attempt - 1))
            else:
                wait_seconds = retry_backoff_seconds * (2 ** (attempt - 1))
            print(
                f"[{source_name}] {request_label} failed ({exc.__class__.__name__}). "
                f"Retry {attempt}/{max_retries} in {wait_seconds:.1f}s"
            )
            time.sleep(wait_seconds)


def list_oai_sets(
    session: requests.Session,
    base_url: str,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> List[Tuple[str, str]]:
    response = _http_get_with_retries(
        session=session,
        url=base_url,
        params={"verb": "ListSets"},
        source_name="list_sets",
        request_label="ListSets",
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as exc:
        snippet = response.content[:500].decode("utf-8", errors="replace")
        print(f"[list_sets] XML ParseError: {exc}. Snippet: {snippet}")
        raise

    sets = []
    for set_el in root.findall(".//oai:set", NS):
        spec = set_el.findtext("oai:setSpec", default="", namespaces=NS)
        name = set_el.findtext("oai:setName", default="", namespaces=NS)
        sets.append((spec, name))
    return sets


def _write_batch_parquet(records: List[Dict], tmp_dir: str, batch_index: int, batch_prefix: str) -> str:
    table = pa.Table.from_pylist(records, schema=pa.schema([
        ("id", pa.string()),
        ("title", pa.string()),
        ("year", pa.int64()),
        ("faculty", pa.string()),
        ("abstract", pa.string()),
        ("source", pa.string()),
        ("url", pa.string()),
    ]))
    file_name = f"{batch_prefix}_{batch_index:04d}.parquet"
    path = os.path.join(tmp_dir, file_name)
    pq.write_table(table, path)
    return path


def _merge_parquet_batches(batch_files: List[str], output_path: str) -> None:
    import duckdb
    if not batch_files:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_parquet(output_path, index=False)
        return

    conn = duckdb.connect()
    # duckdb handles reading a list of files natively
    file_paths = [f"'{p}'" for p in batch_files]
    files_str = ", ".join(file_paths)
    
    # Using QUALIFY for deduplication without pulling everything into RAM
    query = f"""
    COPY (
        SELECT * EXCLUDE (rn) FROM (
            SELECT *, ROW_NUMBER() OVER(PARTITION BY id, url ORDER BY year DESC) as rn 
            FROM read_parquet([{files_str}])
        ) WHERE rn = 1
    ) TO '{output_path}' (FORMAT 'parquet');
    """
    conn.execute(query)



def _next_batch_index(tmp_dir: str, batch_prefix: str) -> int:
    if not os.path.exists(tmp_dir):
        return 1
    existing = [
        name for name in os.listdir(tmp_dir)
        if name.startswith(batch_prefix) and name.endswith(".parquet")
    ]
    if not existing:
        return 1
    indices = []
    for name in existing:
        stem = name.replace(f"{batch_prefix}_", "").replace(".parquet", "")
        if stem.isdigit():
            indices.append(int(stem))
    return max(indices, default=0) + 1


def _parse_date(date_text: str) -> datetime:
    return datetime.strptime(date_text, "%Y-%m-%d")


def build_date_windows(
    from_date: str,
    until_date: Optional[str],
    window_days: int,
) -> List[Tuple[str, str]]:
    start = _parse_date(from_date).date()
    end = _parse_date(until_date).date() if until_date else datetime.utcnow().date()

    windows = []
    current = start
    while current <= end:
        window_end = current
        if window_days > 1:
            window_end = current + timedelta(days=window_days - 1)
        if window_end > end:
            window_end = end
        windows.append((current.isoformat(), window_end.isoformat()))
        current = window_end + timedelta(days=1)
    return windows


def harvest_oai_pmh_batches(
    base_url: str,
    source_name: str,
    metadata_prefix: str = "oai_dc",
    from_date: Optional[str] = None,
    until_date: Optional[str] = None,
    max_records: int = 0,
    pause_seconds: float = 0.2,
    timeout_seconds: int = 60,
    max_retries: int = 3,
    retry_backoff_seconds: float = 1.5,
    session: Optional[requests.Session] = None,
    checkpoint_path: Optional[str] = None,
    tmp_dir: Optional[str] = None,
    batch_prefix: str = "batch",
    window_checkpoint: Optional[Tuple[Optional[str], Optional[str]]] = None,
) -> Tuple[List[str], int, bool]:
    records_collected = 0
    request_count = 0
    batch_files: List[str] = []
    token: Optional[str] = None
    harvest_started_at = _utc_now()

    if session is None:
        session = requests.Session()

    if checkpoint_path:
        if window_checkpoint:
            token, records_collected, harvest_started_at, completed = _get_window_checkpoint(
                checkpoint_path,
                source_name,
                window_checkpoint[0],
                window_checkpoint[1],
            )
            if completed:
                print(
                    f"[{source_name}] Window {window_checkpoint[0]} -> "
                    f"{window_checkpoint[1]} already completed"
                )
                return [], records_collected, True
            if token or records_collected:
                print(
                    f"[{source_name}] checkpoint restored for window "
                    f"{window_checkpoint[0]} -> {window_checkpoint[1]}"
                )
        else:
            state = _load_checkpoint(checkpoint_path)
            if state:
                token = state.get("resumption_token")
                records_collected = state.get("records_collected", 0)
                harvest_started_at = state.get("harvest_started_at", harvest_started_at)
                print(f"[{source_name}] checkpoint restored")

    if tmp_dir:
        ensure_dirs(tmp_dir)
    batch_index = _next_batch_index(tmp_dir or ".", batch_prefix)

    start_time = time.monotonic()

    while True:
        request_count += 1
        if token:
            params = {"verb": "ListRecords", "resumptionToken": token}
            request_label = f"request #{request_count} (resumptionToken)"
        else:
            params = {"verb": "ListRecords", "metadataPrefix": metadata_prefix}
            if from_date:
                params["from"] = from_date
            if until_date:
                params["until"] = until_date
            request_label = f"request #{request_count} (initial)"

        response = _http_get_with_retries(
            session=session,
            url=base_url,
            params=params,
            source_name=source_name,
            request_label=request_label,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )

        response_size = len(response.content) if response.content is not None else 0
        print(
            f"[{source_name}] {request_label} "
            f"URL={response.url} "
            f"Status={response.status_code} "
            f"Size={response_size:,} bytes"
        )

        try:
            batch, token = _parse_oai_streaming(response.content, source_name)
        except ET.ParseError:
            break

        if max_records:
            remaining = max_records - records_collected
            if remaining <= 0:
                break
            if len(batch) > remaining:
                batch = batch[:remaining]

        if batch:
            if tmp_dir:
                batch_path = _write_batch_parquet(batch, tmp_dir, batch_index, batch_prefix)
                batch_files.append(batch_path)
                batch_index += 1
            records_collected += len(batch)

        elapsed = int(time.monotonic() - start_time)
        print(
            f"[{source_name}] batch={request_count} "
            f"records={records_collected} "
            f"elapsed={elapsed}s "
            f"token={'YES' if token else 'END'}"
        )

        if checkpoint_path:
            if window_checkpoint:
                _update_window_checkpoint(
                    checkpoint_path,
                    source_name,
                    window_checkpoint[0],
                    window_checkpoint[1],
                    token,
                    records_collected,
                    harvest_started_at,
                )
            else:
                checkpoint_state = {
                    "source": source_name,
                    "resumption_token": token,
                    "harvest_started_at": harvest_started_at,
                    "last_batch_at": _utc_now(),
                    "records_collected": records_collected,
                }
                _save_checkpoint(checkpoint_path, checkpoint_state)
                print(f"[{source_name}] token saved")

        if max_records and records_collected >= max_records:
            break

        if not token:
            break

        time.sleep(pause_seconds)

    completed = not token
    if checkpoint_path and window_checkpoint:
        _update_window_checkpoint(
            checkpoint_path,
            source_name,
            window_checkpoint[0],
            window_checkpoint[1],
            token,
            records_collected,
            harvest_started_at,
            completed=completed,
        )
        if completed:
            print(
                f"[{source_name}] window {window_checkpoint[0]} -> "
                f"{window_checkpoint[1]} complete"
            )

    return batch_files, records_collected, completed


def harvest_oai_pmh(
    base_url: str,
    source_name: str,
    metadata_prefix: str = "oai_dc",
    from_date: Optional[str] = None,
    until_date: Optional[str] = None,
    max_records: int = 0,
    pause_seconds: float = 0.2,
    timeout_seconds: int = 60,
    max_retries: int = 3,
    retry_backoff_seconds: float = 1.5,
) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp_dir:
        batch_files, _, _ = harvest_oai_pmh_batches(
            base_url=base_url,
            source_name=source_name,
            metadata_prefix=metadata_prefix,
            from_date=from_date,
            until_date=until_date,
            max_records=max_records,
            pause_seconds=pause_seconds,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            session=requests.Session(),
            tmp_dir=tmp_dir,
        )
        table = pa.concat_tables([pq.read_table(path) for path in batch_files]) if batch_files else None
        if table is None:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        return table.to_pandas()


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


def _normalize_text(value: str) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return stripped.lower()


def extract_faculty_from_authorships(authorships: Optional[List[Dict]]) -> str:
    """Extract the most relevant institution/author info from OpenAlex authorships.

    Priority:
    1. UNAM sub-institution (e.g. "Instituto de Ecología") if available.
    2. UNAM author name(s) when only the generic UNAM is listed.
    3. First author's institution as fallback.
    """
    if not authorships:
        return ""

    unam_generic = "Universidad Nacional Autónoma de México"
    unam_sub_institutions = []
    unam_author_names = []
    first_author_inst = ""

    for auth in authorships:
        institutions = auth.get("institutions", [])
        if not institutions:
            continue

        if not first_author_inst:
            first_author_inst = institutions[0].get("display_name", "")

        is_unam = any(
            inst.get("id") in (UNAM_OPENALEX_ID, UNAM_OPENALEX_ID_OLD)
            or "autónoma de méxico" in (inst.get("display_name") or "").lower()
            for inst in institutions
        )

        if is_unam:
            author_name = auth.get("author", {}).get("display_name", "")
            if author_name:
                unam_author_names.append(author_name)

            for inst in institutions:
                name = inst.get("display_name", "")
                if name and name != unam_generic:
                    unam_sub_institutions.append(name)

            raw = auth.get("raw_affiliation_string", "").strip()
            if raw and unam_generic not in raw:
                unam_sub_institutions.append(raw.split(",")[0].strip())

    # Best case: specific UNAM sub-institution
    if unam_sub_institutions:
        return unam_sub_institutions[0]

    # Next best: UNAM author names (up to 2)
    if unam_author_names:
        if len(unam_author_names) <= 2:
            return "; ".join(unam_author_names) + " (UNAM)"
        return f"{unam_author_names[0]}; {unam_author_names[1]} et al. (UNAM)"

    return first_author_inst or ""


def ingest_openalex(
    session: requests.Session,
    base_url: str,
    ror_id: str,
    per_page: int = 200,
    max_records: int = 0,
    timeout_seconds: int = 60,
    max_retries: int = 3,
    retry_backoff_seconds: float = 1.5,
) -> pd.DataFrame:
    records: List[Dict] = []
    cursor = "*"
    request_count = 0

    while True:
        request_count += 1
        params = {
            "filter": f"institutions.ror:{ror_id}",
            "per-page": per_page,
            "cursor": cursor,
        }
        response = _http_get_with_retries(
            session=session,
            url=f"{base_url}/works",
            params=params,
            source_name="openalex",
            request_label=f"request #{request_count}",
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        payload = response.json()

        for work in payload.get("results", []):
            abstract = _abstract_from_inverted_index(work.get("abstract_inverted_index"))
            authorships = work.get("authorships", [])
            landing = (
                work.get("primary_location", {})
                .get("landing_page_url", "")
            )
            faculty = extract_faculty_from_authorships(authorships)
            records.append(
                {
                    "id": work.get("id", ""),
                    "title": work.get("display_name", ""),
                    "year": work.get("publication_year"),
                    "faculty": faculty,
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
    checkpoint_root = os.path.join("data", "checkpoints")
    tmp_root = os.path.join(cfg["paths"]["raw"], "tmp")
    ensure_dirs(
        cfg["paths"]["raw"],
        cfg["paths"]["processed"],
        cfg["paths"]["index"],
        checkpoint_root,
        tmp_root,
    )

    ingest_cfg = cfg.get("ingest", {})
    max_records = ingest_cfg.get("max_records_per_source", 0)
    pause_seconds = ingest_cfg.get("polite_pause_seconds", 0.2)
    timeout_seconds = ingest_cfg.get("request_timeout_seconds", 60)
    max_retries = ingest_cfg.get("request_max_retries", 3)
    retry_backoff_seconds = ingest_cfg.get("retry_backoff_seconds", 1.5)
    continue_on_source_error = ingest_cfg.get("continue_on_source_error", True)
    parallel_time_windows = ingest_cfg.get("parallel_time_windows", False)
    window_days = ingest_cfg.get("window_days", 7)
    max_parallel_workers = ingest_cfg.get("max_parallel_workers", 4)

    session = requests.Session()

    batch_files_all: List[str] = []

    def run_source(label: str, fn):
        try:
            batch_files = fn()
            print(f"[{label}] Collected {len(batch_files):,} batches")
            return batch_files
        except (requests.RequestException, ET.ParseError) as exc:
            if continue_on_source_error:
                print(f"[{label}] Skipped after network errors: {exc}")
                return []
            raise

    if cfg["sources"]["unam_repository"]["enabled"]:
        unam_cfg = cfg["sources"]["unam_repository"]
        checkpoint_path = _checkpoint_path(checkpoint_root, "unam_repository")

        def run_unam():
            if parallel_time_windows and unam_cfg.get("from_date"):
                windows = build_date_windows(
                    unam_cfg["from_date"],
                    unam_cfg.get("until_date"),
                    window_days,
                )
                window_batches: List[str] = []
                with ThreadPoolExecutor(max_workers=max_parallel_workers) as executor:
                    futures = []
                    for win_from, win_until in windows:
                        window_dir = os.path.join(
                            tmp_root,
                            "unam_repository",
                            f"{win_from}_to_{win_until}",
                        )
                        futures.append(executor.submit(
                            harvest_oai_pmh_batches,
                            base_url=unam_cfg["base_url"],
                            source_name="unam_repository",
                            metadata_prefix=unam_cfg["metadata_prefix"],
                            from_date=win_from,
                            until_date=win_until,
                            max_records=max_records,
                            pause_seconds=pause_seconds,
                            timeout_seconds=timeout_seconds,
                            max_retries=max_retries,
                            retry_backoff_seconds=retry_backoff_seconds,
                            session=None,
                            checkpoint_path=checkpoint_path,
                            tmp_dir=window_dir,
                            batch_prefix="batch",
                            window_checkpoint=(win_from, win_until),
                        ))
                    for future in as_completed(futures):
                        batch_files, _, _ = future.result()
                        window_batches.extend(batch_files)

                if _all_windows_completed(checkpoint_path, windows):
                    _delete_checkpoint(checkpoint_path)
                return window_batches

            tmp_dir = os.path.join(tmp_root, "unam_repository")
            batch_files, _, completed = harvest_oai_pmh_batches(
                base_url=unam_cfg["base_url"],
                source_name="unam_repository",
                metadata_prefix=unam_cfg["metadata_prefix"],
                from_date=unam_cfg.get("from_date"),
                until_date=unam_cfg.get("until_date"),
                max_records=max_records,
                pause_seconds=pause_seconds,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                session=session,
                checkpoint_path=checkpoint_path,
                tmp_dir=tmp_dir,
                batch_prefix="batch",
            )
            if completed:
                _delete_checkpoint(checkpoint_path)
            return batch_files

        batch_files_all.extend(run_source("unam_repository", run_unam))

    if cfg["sources"]["scielo"]["enabled"]:
        scielo_cfg = cfg["sources"]["scielo"]
        checkpoint_path = _checkpoint_path(checkpoint_root, "scielo")

        def run_scielo():
            if parallel_time_windows and scielo_cfg.get("from_date"):
                windows = build_date_windows(
                    scielo_cfg["from_date"],
                    scielo_cfg.get("until_date"),
                    window_days,
                )
                window_batches: List[str] = []
                with ThreadPoolExecutor(max_workers=max_parallel_workers) as executor:
                    futures = []
                    for win_from, win_until in windows:
                        window_dir = os.path.join(
                            tmp_root,
                            "scielo",
                            f"{win_from}_to_{win_until}",
                        )
                        futures.append(executor.submit(
                            harvest_oai_pmh_batches,
                            base_url=scielo_cfg["base_url"],
                            source_name="scielo_mexico",
                            metadata_prefix=scielo_cfg["metadata_prefix"],
                            from_date=win_from,
                            until_date=win_until,
                            max_records=max_records,
                            pause_seconds=pause_seconds,
                            timeout_seconds=timeout_seconds,
                            max_retries=max_retries,
                            retry_backoff_seconds=retry_backoff_seconds,
                            session=None,
                            checkpoint_path=checkpoint_path,
                            tmp_dir=window_dir,
                            batch_prefix="batch",
                            window_checkpoint=(win_from, win_until),
                        ))
                    for future in as_completed(futures):
                        batch_files, _, _ = future.result()
                        window_batches.extend(batch_files)

                if _all_windows_completed(checkpoint_path, windows):
                    _delete_checkpoint(checkpoint_path)
                return window_batches

            tmp_dir = os.path.join(tmp_root, "scielo")
            batch_files, _, completed = harvest_oai_pmh_batches(
                base_url=scielo_cfg["base_url"],
                source_name="scielo_mexico",
                metadata_prefix=scielo_cfg["metadata_prefix"],
                from_date=scielo_cfg.get("from_date"),
                until_date=scielo_cfg.get("until_date"),
                max_records=max_records,
                pause_seconds=pause_seconds,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                session=session,
                checkpoint_path=checkpoint_path,
                tmp_dir=tmp_dir,
                batch_prefix="batch",
            )
            if completed:
                _delete_checkpoint(checkpoint_path)
            return batch_files

        batch_files_all.extend(run_source("scielo_mexico", run_scielo))

    if cfg["sources"]["openalex"]["enabled"]:
        def run_openalex():
            df = ingest_openalex(
                session=session,
                base_url=cfg["sources"]["openalex"]["base_url"],
                ror_id=cfg["sources"]["openalex"]["ror_id"],
                per_page=cfg["sources"]["openalex"]["per_page"],
                max_records=max_records,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            tmp_dir = os.path.join(tmp_root, "openalex")
            ensure_dirs(tmp_dir)
            if df.empty:
                return []
            import numpy as np
            records = df.replace({np.nan: None}).to_dict("records")
            batch_path = _write_batch_parquet(records, tmp_dir, 1, "batch")
            return [batch_path]

        batch_files_all.extend(run_source("openalex", run_openalex))

    out_path = os.path.join(cfg["paths"]["raw"], "records.parquet")
    _merge_parquet_batches(batch_files_all, out_path)
    print(f"Saved: {out_path}")

    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--list-sets",
        choices=["unam_repository", "scielo"],
        help="List OAI-PMH sets for a source and exit",
    )
    args = parser.parse_args()
    if args.list_sets:
        cfg = load_config(args.config)
        source_cfg = cfg["sources"].get(args.list_sets)
        if not source_cfg:
            raise ValueError(f"Unknown source: {args.list_sets}")
        ingest_cfg = cfg.get("ingest", {})
        session = requests.Session()
        sets = list_oai_sets(
            session=session,
            base_url=source_cfg["base_url"],
            timeout_seconds=ingest_cfg.get("request_timeout_seconds", 60),
            max_retries=ingest_cfg.get("request_max_retries", 3),
            retry_backoff_seconds=ingest_cfg.get("retry_backoff_seconds", 1.5),
        )
        for spec, name in sets:
            print(f"{spec}\t{name}")
    else:
        main(args.config)
