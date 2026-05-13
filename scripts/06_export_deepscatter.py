"""
Step 06: Export dataset to DeepScatter quadfeather tiles.
Reads data/processed/unam_embeddings_2d.parquet and outputs to data/tiles/
"""
import argparse
import os
import subprocess
from semantic_research_atlas.utils import load_config, ensure_dirs


def main(config_path: str):
    cfg = load_config(config_path)
    
    in_path = f"{cfg['paths']['processed']}/unam_embeddings_2d.parquet"
    out_dir = "data/tiles"
    
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}. Run previous steps.")
    
    ensure_dirs(out_dir)
    
    import pyarrow.parquet as pq
    import pyarrow as pa

    print("Downcasting LargeUtf8 to Utf8 and Int64 to Int32 for DeepScatter compatibility...")
    table = pq.read_table(in_path)
    new_schema = []
    for field in table.schema:
        if field.type == pa.large_string():
            new_schema.append(pa.field(field.name, pa.string()))
        elif field.type == pa.int64():
            new_schema.append(pa.field(field.name, pa.int32()))
        else:
            new_schema.append(field)
            
    casted_table = table.cast(pa.schema(new_schema))
    temp_path = in_path.replace(".parquet", "_temp.parquet")
    pq.write_table(casted_table, temp_path)
    
    print(f"Exporting {temp_path} to DeepScatter tiles at {out_dir}...")
    
    cmd = [
        "quadfeather",
        "--files", temp_path,
        "--destination", out_dir
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("Tile generation complete! Post-processing to inject metadata...")
        
        # DeepScatter expects metadata (extent, children) inside the actual .feather files
        import pyarrow.feather as feather
        import json
        manifest_path = os.path.join(out_dir, "manifest.feather")
        if os.path.exists(manifest_path):
            manifest_df = feather.read_table(manifest_path).to_pandas()
            for _, row in manifest_df.iterrows():
                key = row['key']
                path = os.path.join(out_dir, f"{key}.feather")
                if not os.path.exists(path): continue
                
                table = feather.read_table(path)
                extent_str = row['extent']
                
                # find children
                z, x, y = map(int, key.split('/'))
                children = []
                for i in range(4):
                    cx = x * 2 + (i % 2)
                    cy = y * 2 + (i // 2)
                    child_key = f"{z+1}/{cx}/{cy}"
                    if child_key in manifest_df['key'].values:
                        children.append(child_key)
                
                metadata = {}
                if extent_str:
                    metadata[b'extent'] = extent_str.encode('utf-8')
                if children:
                    metadata[b'children'] = json.dumps(children).encode('utf-8')
                    
                if metadata:
                    existing_meta = table.schema.metadata or {}
                    existing_meta.update(metadata)
                    new_schema = table.schema.with_metadata(existing_meta)
                    new_table = table.cast(new_schema)
                    feather.write_feather(new_table, path, compression='uncompressed')
            print("Post-processing complete!")
        else:
            print("Warning: manifest.feather not found.")
    except subprocess.CalledProcessError as e:
        print(f"Error generating tiles: {e}")
        raise
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
