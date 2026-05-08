# Semantic Research Atlas

Semantic clustering and interactive exploration of academic production from UNAM sources plus OpenAlex and SciELO. The project follows the same end-to-end flow as **jpbascur/text-similarity-maker**, adding a **Self-Organizing Map (SOM)** view for comparison.

## ✨ Features
- **Ingestion** from: UNAM Institutional Repository, OpenAlex, SciELO (configurable)
- **Embeddings** with multilingual models (Spanish-friendly)
- **Clustering** (UMAP + HDBSCAN)
- **SOM** view (U-Matrix + neuron mapping)
- **Interactive app** (Streamlit) with:
  - UMAP map (WebGL)
  - SOM grid view
  - Semantic search (FAISS)

## 📁 Repository Structure
```
app/                  # Streamlit app
config/               # YAML configs (sources, models, params)
notebooks/            # Optional experiments
scripts/              # CLI pipeline steps
src/semantic_research_atlas/  # Library code
```

## 🚀 Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🧩 Pipeline (Local/Colab)
1. **Ingest**
```bash
python scripts/01_ingest.py --config config/default.yaml
```
2. **Embed**
```bash
python scripts/02_embed.py --config config/default.yaml
```
3. **UMAP + Clustering**
```bash
python scripts/03_umap_cluster.py --config config/default.yaml
```
4. **SOM**
```bash
python scripts/04_som.py --config config/default.yaml
```
5. **Build FAISS Index**
```bash
python scripts/05_faiss.py --config config/default.yaml
```

## 📊 App
```bash
streamlit run app/streamlit_app.py
```

## 📦 Output Artifacts
- `data/processed/unam_embeddings_2d.parquet`
- `data/processed/som_map.parquet`
- `data/index/index.faiss`

## 📌 Notes
- For 300k records, **Parquet + FAISS** is recommended.
- For SOM, train on a **subsample (e.g., 50k)** then map the full set.

## 🔧 Configuration
All settings live in `config/default.yaml`.

---

If you want this deployed to **Hugging Face Spaces**, Streamlit works out-of-the-box.
