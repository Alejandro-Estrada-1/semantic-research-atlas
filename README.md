# Semantic Research Atlas

A highly scalable, 100% local, browser-controlled pipeline to construct interactive WebGL semantic maps of any institution's scholarly production in the world.

![Semantic Research Atlas Overview](frontend/public/atlas.png)

> [!NOTE]
> **Privacy First**: This architecture runs entirely on your local machine. No data is sent to external servers for embedding or clustering.

## What is this?
This project transforms raw metadata from millions of academic papers into a fully interactive topography (knowledge map). It allows you to:
1. Search for any institution worldwide (e.g. "MIT", "UNAM", "Harvard") via OpenAlex.
2. Build a local AI-powered pipeline to vectorize their scientific output.
3. Visualize the documents natively in your browser using **DeepScatter** (WebGL) and **Self-Organizing Maps (SOM)**.

## Key Features
- **Universal Search**: Powered by OpenAlex Autocomplete. Type any institution and start building.
- **Extreme Local Performance**: Uses CTranslate2 + INT8 Quantization + AVX2 to process thousands of embeddings per second strictly on your CPU. No Colab, no cloud GPUs needed.
- **Zero-RAM Pipeline**: Uses DuckDB for chunked batching. Even datasets with 1M+ records can be built on 8GB RAM laptops.
- **Browser Controlled**: Launch, monitor, and abort the data pipeline directly from the React UI.

> [!TIP]
> **Fast Pipeline Mode**: If you just want to test the waters, choose the "Fast Pipeline (15k max)" mode in the UI. It restricts the ingestion to the most recent 15,000 papers and generates a full interactive Atlas in ~2 minutes!

## Architecture & Technology Stack
- **Ingestion & Data**: OpenAlex API, DuckDB, PyArrow (Parquet)
- **AI Models**: CTranslate2 (INT8 CPU Inference), Sentence-Transformers (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Clustering & Topography**: UMAP, HDBSCAN, MiniSom, TF-IDF
- **Frontend & Rendering**: React 19, Vite, DeepScatter (Quadfeather), HTML5 Canvas
- **Orchestration**: Flask + Subprocess management

---

## Installation

### 1. Prerequisites
- Python 3.10+
- Node.js 18+

### 2. Python Backend (Virtual Environment)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> [!IMPORTANT]
> **Crucial for Speed**: To download metadata effectively without being blocked by rate limits, you **must use your own OpenAlex API Key**.

### 3. OpenAlex API Key Setup
1. Create a free account and get your personal API Key at: [openalex.org/settings/api](https://openalex.org/settings/api)
2. In the root folder of this project, create a new file named exactly `.env`
3. Paste your personal key inside that file like this:

```env
OPENALEX_API_KEY=your_personal_key_here
```

> [!CAUTION]
> **Security Warning**: Never commit your `.env` file to a public repository. The `.gitignore` in this project is already configured to ignore it, but always double-check before committing if you use a different version control system.

### 4. Node Frontend
```bash
cd frontend
npm install
cd ..
```

---

## Running the Atlas

> [!WARNING]
> **Heavy Processing**: Running the "Full Pipeline" on massive universities (like Harvard or MIT) can take hours and will utilize all available CPU cores. Make sure your laptop is plugged into power!

You need to run both the API Server and the Frontend Development Server.

**Terminal 1: Flask API Server**
```bash
source venv/bin/activate
python server.py
# Server runs on http://localhost:8000
```

**Terminal 2: React Frontend**
```bash
cd frontend
npm run dev
# App runs on http://localhost:5173
```

Open `http://localhost:5173` in your browser. Use the search bar to find an institution, choose a build mode ("Fast" or "Full"), and watch the semantic map being constructed live!
