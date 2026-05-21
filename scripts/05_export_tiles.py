"""
Step 05: Export dataset to DeepScatter quadfeather tiles.

Usage:
    python scripts/05_export_tiles.py --config config/default.yaml --inst-id 01tmp8f25 --mode full

Outputs: data/tiles/{inst_id}_{mode}/ (quadfeather .feather files)
"""
import argparse
import json
import os
import subprocess

import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq

from semantic_research_atlas.utils import load_config, ensure_dirs


def main(config_path: str, inst_id: str = "default", mode: str = "full"):
    cfg = load_config(config_path)

    prefix = f"{inst_id}_{mode}"
    proc_dir = os.path.join(cfg["paths"]["processed"], prefix)
    tiles_dir = os.path.join("data/tiles", prefix)

    in_path = os.path.join(proc_dir, "embeddings_2d.parquet")
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Not found: {in_path}. Run 03_umap_cluster.py first.")

    ensure_dirs(tiles_dir)

    # ── Downcast types for DeepScatter compatibility ──
    print("Downcasting types for DeepScatter...")
    table = pq.read_table(in_path)
    new_fields = []
    for field in table.schema:
        if field.type == pa.large_string():
            new_fields.append(pa.field(field.name, pa.string()))
        elif field.type == pa.int64():
            new_fields.append(pa.field(field.name, pa.int32()))
        else:
            new_fields.append(field)

    casted = table.cast(pa.schema(new_fields))
    temp_path = os.path.join(proc_dir, "embeddings_2d_temp.parquet")
    pq.write_table(casted, temp_path)

    # ── Generate tiles ──
    print(f"Generating tiles at {tiles_dir}...")
    cmd = ["quadfeather", "--files", temp_path, "--destination", tiles_dir]

    try:
        subprocess.run(cmd, check=True)
        print("Tile generation complete. Post-processing metadata...")

        manifest_path = os.path.join(tiles_dir, "manifest.feather")
        if os.path.exists(manifest_path):
            manifest_df = feather.read_table(manifest_path).to_pandas()
            for _, row in manifest_df.iterrows():
                key = row["key"]
                tile_path = os.path.join(tiles_dir, f"{key}.feather")
                if not os.path.exists(tile_path):
                    continue

                tile_table = feather.read_table(tile_path)
                extent_str = row["extent"]

                z, x, y = map(int, key.split("/"))
                children = []
                for i in range(4):
                    child_key = f"{z+1}/{x*2 + i%2}/{y*2 + i//2}"
                    if child_key in manifest_df["key"].values:
                        children.append(child_key)

                meta = {}
                if extent_str:
                    meta[b"extent"] = extent_str.encode("utf-8")
                if children:
                    meta[b"children"] = json.dumps(children).encode("utf-8")

                if meta:
                    existing = tile_table.schema.metadata or {}
                    existing.update(meta)
                    new_schema = tile_table.schema.with_metadata(existing)
                    feather.write_feather(
                        tile_table.cast(new_schema), tile_path, compression="uncompressed"
                    )
            print("✓ Post-processing complete!")
        else:
            print("⚠ manifest.feather not found.")
    except subprocess.CalledProcessError as e:
        print(f"✗ Tile generation failed: {e}")
        raise
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print(f"✓ Tiles ready: {tiles_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--inst-id", default="default")
    parser.add_argument("--mode", default="full", choices=["full", "limited"])
    args = parser.parse_args()
    main(args.config, args.inst_id, args.mode)
