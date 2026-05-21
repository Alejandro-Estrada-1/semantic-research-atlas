"""
Semantic Research Atlas — API Server & Pipeline Orchestrator.

Replaces the static serve_cors.py with a full Flask API that:
- Proxies OpenAlex autocomplete (bypasses CORS)
- Serves tile data with CORS headers
- Orchestrates the pipeline (01→06) via subprocess
- Streams live progress and supports abort

Usage:
    python server.py                          # default port 8000
    python server.py --port 9000              # custom port
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

# ── Global pipeline state (one pipeline at a time) ──────────────────────────

pipeline_state = {
    "running": False,
    "step": "",
    "step_num": 0,
    "total_steps": 6,
    "progress": "",
    "log_lines": [],
    "process": None,
    "inst_id": "",
    "inst_name": "",
    "mode": "",
    "started_at": None,
    "error": None,
    "completed": False,
}

_state_lock = threading.Lock()
MAX_LOG_LINES = 200


def _reset_state():
    """Reset pipeline state to idle."""
    pipeline_state.update({
        "running": False,
        "step": "",
        "step_num": 0,
        "progress": "",
        "log_lines": [],
        "process": None,
        "inst_id": "",
        "inst_name": "",
        "mode": "",
        "started_at": None,
        "error": None,
        "completed": False,
    })


# ── Pipeline runner ─────────────────────────────────────────────────────────

PIPELINE_STEPS = [
    ("ingest",  "01_ingest.py",         "Ingesting metadata from OpenAlex"),
    ("embed",   "02_embed.py",          "Generating INT8 embeddings"),
    ("umap",    "03_umap_cluster.py",   "UMAP + HDBSCAN clustering"),
    ("som",     "04_som.py",            "Training Self-Organizing Map"),
    ("tiles",   "05_export_tiles.py",   "Exporting DeepScatter tiles"),
    ("labels",  "06_generate_labels.py", "Generating semantic labels"),
]


def _read_output(proc, step_label):
    """Read stdout/stderr from subprocess line by line into pipeline_state."""
    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            line = line.strip()
            if line:
                with _state_lock:
                    pipeline_state["progress"] = line
                    pipeline_state["log_lines"].append(f"[{step_label}] {line}")
                    if len(pipeline_state["log_lines"]) > MAX_LOG_LINES:
                        pipeline_state["log_lines"] = pipeline_state["log_lines"][-MAX_LOG_LINES:]
    except (ValueError, OSError):
        pass  # pipe closed


def _run_pipeline(inst_id, inst_name, filter_key, filter_value, max_records, mode):
    """Execute pipeline scripts 01→06 sequentially in a background thread."""
    config_path = "config/default.yaml"

    try:
        for i, (step_key, script, description) in enumerate(PIPELINE_STEPS):
            with _state_lock:
                if not pipeline_state["running"]:
                    return  # aborted
                pipeline_state["step"] = step_key
                pipeline_state["step_num"] = i + 1
                pipeline_state["progress"] = description
                pipeline_state["log_lines"].append(f"{'─'*40}")
                pipeline_state["log_lines"].append(f"Step {i+1}/6: {description}")

            # Build command arguments
            cmd = [
                sys.executable, f"scripts/{script}",
                "--config", config_path,
                "--inst-id", inst_id,
                "--mode", mode,
            ]

            # Script-specific args
            if script == "01_ingest.py":
                cmd += [
                    "--filter-key", filter_key,
                    "--filter-value", filter_value,
                    "--max-records", str(max_records),
                ]

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

            with _state_lock:
                pipeline_state["process"] = proc

            # Read output in this thread (we're already in a background thread)
            _read_output(proc, step_key)
            proc.wait()

            with _state_lock:
                pipeline_state["process"] = None

            if proc.returncode != 0:
                raise RuntimeError(f"Step {script} failed with exit code {proc.returncode}")

        # ── All steps completed ──
        with _state_lock:
            pipeline_state["running"] = False
            pipeline_state["completed"] = True
            pipeline_state["step"] = "done"
            pipeline_state["progress"] = "Pipeline completed successfully!"
            pipeline_state["log_lines"].append(f"{'='*40}")
            pipeline_state["log_lines"].append("✓ Atlas ready! Switch to the Atlas view.")

    except Exception as e:
        with _state_lock:
            pipeline_state["running"] = False
            pipeline_state["error"] = str(e)
            pipeline_state["progress"] = f"Error: {e}"
            pipeline_state["log_lines"].append(f"✗ ERROR: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════════════════════

# ── OpenAlex Proxy (bypasses CORS for the frontend) ─────────────────────────

@app.route("/api/search/institutions")
def search_institutions():
    q = request.args.get("q", "")
    if len(q) < 2:
        return jsonify({"results": []})

    params = {"q": q}
    api_key = os.environ.get("OPENALEX_API_KEY")
    if api_key:
        params["api_key"] = api_key

    try:
        resp = requests.get(
            "https://api.openalex.org/autocomplete/institutions",
            params=params, timeout=10,
        )
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e), "results": []}), 502


@app.route("/api/search/global")
def search_global():
    q = request.args.get("q", "")
    if len(q) < 2:
        return jsonify({"results": []})

    params = {"q": q}
    api_key = os.environ.get("OPENALEX_API_KEY")
    if api_key:
        params["api_key"] = api_key

    try:
        resp = requests.get(
            "https://api.openalex.org/autocomplete",
            params=params, timeout=10,
        )
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e), "results": []}), 502


# ── Pipeline control ────────────────────────────────────────────────────────

@app.route("/api/pipeline/check/<inst_id>")
def pipeline_check(inst_id):
    """Check what tiles already exist on disk for this institution."""
    available_modes = []
    for m in ["limited", "full"]:
        tiles_dir = os.path.join("data/tiles", f"{inst_id}_{m}")
        if os.path.exists(os.path.join(tiles_dir, "0/0/0.feather")) and os.path.exists(os.path.join(tiles_dir, "cluster_labels.json")):
            available_modes.append(m)

    return jsonify({
        "exists": len(available_modes) > 0,
        "available_modes": available_modes
    })


@app.route("/api/pipeline/start", methods=["POST"])
def pipeline_start():
    """Start the full pipeline (01→06) as a background process."""
    with _state_lock:
        if pipeline_state["running"]:
            return jsonify({"error": "Pipeline already running"}), 409

    data = request.json or {}
    inst_id = data.get("inst_id", "")
    inst_name = data.get("inst_name", "")
    filter_key = data.get("filter_key", "ror")
    filter_value = data.get("filter_value", "")
    max_records = int(data.get("max_records", 0))
    mode = data.get("mode", "full")

    if not inst_id or not filter_value:
        return jsonify({"error": "inst_id and filter_value are required"}), 400

    with _state_lock:
        _reset_state()
        pipeline_state["running"] = True
        pipeline_state["inst_id"] = inst_id
        pipeline_state["inst_name"] = inst_name
        pipeline_state["mode"] = mode
        pipeline_state["started_at"] = datetime.now(timezone.utc).isoformat()

    thread = threading.Thread(
        target=_run_pipeline,
        args=(inst_id, inst_name, filter_key, filter_value, max_records, mode),
        daemon=True,
    )
    thread.start()

    return jsonify({"status": "started", "inst_id": inst_id, "mode": mode})


@app.route("/api/pipeline/status")
def pipeline_status():
    """Return current pipeline state and recent log lines."""
    with _state_lock:
        return jsonify({
            "running": pipeline_state["running"],
            "completed": pipeline_state["completed"],
            "step": pipeline_state["step"],
            "step_num": pipeline_state["step_num"],
            "total_steps": pipeline_state["total_steps"],
            "progress": pipeline_state["progress"],
            "inst_id": pipeline_state["inst_id"],
            "inst_name": pipeline_state["inst_name"],
            "mode": pipeline_state["mode"],
            "started_at": pipeline_state["started_at"],
            "error": pipeline_state["error"],
            "log_lines": pipeline_state["log_lines"][-50:],
        })


@app.route("/api/pipeline/abort", methods=["POST"])
def pipeline_abort():
    """Abort the running pipeline by terminating the subprocess."""
    with _state_lock:
        if not pipeline_state["running"]:
            return jsonify({"status": "not_running"})

        proc = pipeline_state["process"]
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        pipeline_state["running"] = False
        pipeline_state["error"] = "Aborted by user"
        pipeline_state["progress"] = "Pipeline aborted."
        pipeline_state["log_lines"].append("⚠ Pipeline aborted by user.")

    return jsonify({"status": "aborted"})


# ── Static file serving (tiles, JSON data) ──────────────────────────────────

@app.route("/data/<path:filepath>")
def serve_data(filepath):
    """Serve files from the data/ directory with CORS headers."""
    return send_from_directory("data", filepath)


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Research Atlas API Server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"╔{'═'*56}╗")
    print(f"║  Semantic Research Atlas — API Server                ║")
    print(f"║  http://localhost:{args.port:<5}                            ║")
    api_key = os.environ.get("OPENALEX_API_KEY")
    if api_key:
        print(f"║  OpenAlex API Key: {api_key[:8]}...{'':>24}║")
    else:
        print(f"║  ⚠ No OPENALEX_API_KEY in .env (low rate limit)      ║")
    print(f"╚{'═'*56}╝")

    app.run(host="0.0.0.0", port=args.port, debug=args.debug, threaded=True)
