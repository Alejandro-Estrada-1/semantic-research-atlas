"""
Step 01: Ingest metadata from OpenAlex for any institution worldwide.

Usage:
    python scripts/01_ingest.py --config config/default.yaml \\
        --inst-id 01tmp8f25 --filter-key ror --filter-value 01tmp8f25 \\
        --mode full --max-records 0

Outputs: data/raw/{inst_id}_{mode}/records.parquet
"""
import argparse
import os
import shutil
import tempfile
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from dotenv import load_dotenv

from semantic_research_atlas.utils import load_config, ensure_dirs

load_dotenv()

OUTPUT_COLUMNS = ["id", "title", "year", "faculty", "abstract", "source", "url"]
RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def _http_get_with_retries(
    session: requests.Session,
    url: str,
    params: Dict,
    request_label: str,
    timeout_seconds: int = 120,
    max_retries: int = 6,
    retry_backoff_seconds: float = 3.0,
) -> requests.Response:
    """Resilient HTTP GET with exponential backoff and Retry-After support."""
    attempt = 0
    headers = {
        "User-Agent": "semantic-research-atlas/1.0 (research metadata harvester)",
        "Accept": "application/json",
    }
    # Inject API key if available
    api_key = os.environ.get("OPENALEX_API_KEY")
    if api_key:
        params["api_key"] = api_key

    while True:
        try:
            response = session.get(url, params=params, timeout=timeout_seconds, headers=headers)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code is not None and status_code not in RETRYABLE_STATUS_CODES:
                raise
            if attempt >= max_retries:
                raise

            attempt += 1
            resp = getattr(exc, "response", None)
            retry_after = resp.headers.get("Retry-After") if resp is not None else None

            if retry_after:
                try:
                    wait = float(retry_after)
                except ValueError:
                    wait = retry_backoff_seconds * (2 ** (attempt - 1))
            else:
                wait = retry_backoff_seconds * (2 ** (attempt - 1))

            print(f"  ⚠ {request_label} failed ({exc.__class__.__name__}). "
                  f"Retry {attempt}/{max_retries} in {wait:.1f}s")
            time.sleep(wait)


# ── Parquet batch I/O ────────────────────────────────────────────────────────

def _write_batch_parquet(records: List[Dict], tmp_dir: str, batch_index: int) -> str:
    """Write a batch of records to a numbered Parquet file."""
    table = pa.Table.from_pylist(records, schema=pa.schema([
        ("id", pa.string()),
        ("title", pa.string()),
        ("year", pa.int64()),
        ("faculty", pa.string()),
        ("abstract", pa.string()),
        ("source", pa.string()),
        ("url", pa.string()),
    ]))
    path = os.path.join(tmp_dir, f"batch_{batch_index:04d}.parquet")
    pq.write_table(table, path)
    return path


def _merge_parquet_batches(batch_files: List[str], output_path: str) -> None:
    """Merge and deduplicate batches via DuckDB (zero RAM overhead)."""
    import duckdb

    if not batch_files:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_parquet(output_path, index=False)
        return

    conn = duckdb.connect()
    files_str = ", ".join(f"'{p}'" for p in batch_files)
    conn.execute(f"""
        COPY (
            SELECT * EXCLUDE (rn) FROM (
                SELECT *, ROW_NUMBER() OVER(PARTITION BY id ORDER BY year DESC NULLS LAST) as rn
                FROM read_parquet([{files_str}])
            ) WHERE rn = 1
        ) TO '{output_path}' (FORMAT 'parquet');
    """)
    conn.close()


# ── OpenAlex abstract reconstruction ────────────────────────────────────────

def _abstract_from_inverted_index(inv: Optional[Dict]) -> str:
    """Reconstruct plaintext abstract from OpenAlex's inverted index format."""
    if not inv:
        return ""
    max_pos = max(pos for positions in inv.values() for pos in positions)
    words = [""] * (max_pos + 1)
    for word, positions in inv.items():
        for pos in positions:
            if pos < len(words):
                words[pos] = word
    return " ".join(w for w in words if w)


def _extract_faculty(authorships: Optional[List[Dict]]) -> str:
    """Extract the primary institution name from the first author's affiliations."""
    if not authorships:
        return ""
    for auth in authorships:
        institutions = auth.get("institutions", [])
        if institutions:
            return institutions[0].get("display_name", "")
    return ""


# ── Main OpenAlex ingestion ─────────────────────────────────────────────────

def ingest_openalex(
    session: requests.Session,
    filter_key: str,
    filter_value: str,
    per_page: int = 200,
    max_records: int = 0,
    timeout_seconds: int = 120,
    max_retries: int = 6,
    retry_backoff_seconds: float = 3.0,
    tmp_dir: str = "",
) -> List[str]:
    """
    Paginate through OpenAlex /works endpoint and write batches to disk.

    Args:
        filter_key: "ror" or "openalex_id" — determines the filter parameter.
        filter_value: The actual ID value to filter by.
    Returns:
        List of batch file paths.
    """
    batch_files: List[str] = []
    batch_index = 1
    cursor = "*"
    request_count = 0
    records_buffer: List[Dict] = []
    total_collected = 0
    start = time.time()

    # Dynamic filter based on whether we have a ROR or OpenAlex ID
    if filter_key == "ror":
        filter_param = f"institutions.ror:{filter_value}"
    else:
        filter_param = f"institutions.id:{filter_value}"

    while True:
        request_count += 1
        params = {
            "filter": filter_param,
            "per-page": per_page,
            "cursor": cursor,
            "select": "id,display_name,publication_year,authorships,abstract_inverted_index,primary_location",
        }

        response = _http_get_with_retries(
            session=session,
            url="https://api.openalex.org/works",
            params=params,
            request_label=f"page {request_count}",
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        payload = response.json()

        # First page: log total count
        if request_count == 1:
            api_total = payload.get("meta", {}).get("count", 0)
            effective = f"{max_records:,}" if max_records else f"{api_total:,} (all)"
            print(f"  OpenAlex reports {api_total:,} works. Downloading {effective}...")

        for work in payload.get("results", []):
            abstract = _abstract_from_inverted_index(work.get("abstract_inverted_index"))
            faculty = _extract_faculty(work.get("authorships"))
            landing = (work.get("primary_location") or {}).get("landing_page_url", "")

            records_buffer.append({
                "id": work.get("id", ""),
                "title": work.get("display_name", ""),
                "year": work.get("publication_year"),
                "faculty": faculty,
                "abstract": abstract,
                "source": "openalex",
                "url": landing or work.get("id", ""),
            })

            total_collected += 1

            if max_records and total_collected >= max_records:
                break

        # Flush buffer to disk every 5000 records
        if len(records_buffer) >= 5000:
            clean = [
                {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}
                for r in records_buffer
            ]
            path = _write_batch_parquet(clean, tmp_dir, batch_index)
            batch_files.append(path)
            batch_index += 1
            records_buffer.clear()

        elapsed = time.time() - start
        rate = total_collected / elapsed if elapsed > 0 else 0
        print(f"  [page {request_count}] {total_collected:,} records | {rate:.0f} docs/s", end="\r")

        if max_records and total_collected >= max_records:
            break

        cursor = payload.get("meta", {}).get("next_cursor")
        if not cursor:
            break

    # Flush remaining records
    if records_buffer:
        clean = [
            {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}
            for r in records_buffer
        ]
        path = _write_batch_parquet(clean, tmp_dir, batch_index)
        batch_files.append(path)

    print(f"\n  ✓ Downloaded {total_collected:,} records in {len(batch_files)} batches")
    return batch_files


# ── CLI entrypoint ──────────────────────────────────────────────────────────

def main(
    config_path: str,
    inst_id: str,
    filter_key: str,
    filter_value: str,
    max_records: int,
    mode: str,
):
    cfg = load_config(config_path)
    ingest_cfg = cfg.get("ingest", {})

    prefix = f"{inst_id}_{mode}"
    out_dir = os.path.join(cfg["paths"]["raw"], prefix)
    ensure_dirs(out_dir)

    out_path = os.path.join(out_dir, "records.parquet")

    print(f"{'='*60}")
    print(f"Ingesting: {inst_id} ({filter_key}={filter_value})")
    print(f"Mode: {mode} | Max records: {max_records or 'unlimited'}")
    print(f"Output: {out_path}")
    print(f"{'='*60}")

    if os.path.exists(out_path):
        import duckdb
        conn = duckdb.connect()
        try:
            count = conn.execute(f"SELECT count(*) FROM '{out_path}'").fetchone()[0]
            if count > 0:
                print(f" File already exists with {count:,} records. Skipping download.")
                conn.close()
                return
        except Exception:
            pass
        finally:
            conn.close()

    with tempfile.TemporaryDirectory() as tmp_dir:
        session = requests.Session()

        batch_files = ingest_openalex(
            session=session,
            filter_key=filter_key,
            filter_value=filter_value,
            per_page=ingest_cfg.get("per_page", 200),
            max_records=max_records,
            timeout_seconds=ingest_cfg.get("request_timeout_seconds", 120),
            max_retries=ingest_cfg.get("request_max_retries", 6),
            retry_backoff_seconds=ingest_cfg.get("retry_backoff_seconds", 3),
            tmp_dir=tmp_dir,
        )

        _merge_parquet_batches(batch_files, out_path)

    print(f" Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest metadata from OpenAlex")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--inst-id", required=True, help="Institution ID (ROR or OpenAlex short ID)")
    parser.add_argument("--filter-key", required=True, choices=["ror", "openalex_id"],
                        help="Filter type: 'ror' for ROR ID, 'openalex_id' for OpenAlex native ID")
    parser.add_argument("--filter-value", required=True, help="The ID value to filter by")
    parser.add_argument("--max-records", type=int, default=0, help="Max records (0 = unlimited)")
    parser.add_argument("--mode", default="full", choices=["full", "limited"])
    args = parser.parse_args()
    main(args.config, args.inst_id, args.filter_key, args.filter_value, args.max_records, args.mode)
