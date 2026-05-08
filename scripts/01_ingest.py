"""
Step 01: Ingest metadata from UNAM repository, OpenAlex, SciELO.
Outputs a unified parquet in data/raw/records.parquet
"""
import argparse
import pandas as pd
from semantic_research_atlas.utils import load_config, ensure_dirs


def main(config_path: str):
    cfg = load_config(config_path)
    ensure_dirs(cfg["paths"]["raw"], cfg["paths"]["processed"], cfg["paths"]["index"])

    # TODO: implement actual ingestion for each source
    # Placeholder schema
    df = pd.DataFrame(
        columns=["id", "title", "year", "faculty", "abstract", "source", "url"]
    )

    out_path = f"{cfg['paths']['raw']}/records.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
