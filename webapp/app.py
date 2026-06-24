"""
Premium Flask web app for Traffic Demand Prediction.

Phase 4: wired to the persisted 5-model XGBStack ensemble in
``output/models/`` via ``src/inference.py``. Predictions are real demand
values in [0, 1].

The selected ensemble method is ``XGBStack``: at inference we average 5
fitted stacker boosters over the per-algo averaged base-model predictions.
``ensemble_meta["weights"]`` is unused (equal priors — would be a
meaningless linear blend for XGBStack).
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Backwards compatibility for unpickling old models
import src.features.feature_engineering
sys.modules['src.feature_engineering'] = src.features.feature_engineering

from flask import Flask, jsonify, render_template, request, send_file

from src.models.inference import load_artifacts, predict_batch, predict_single

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB
app.config["JSON_SORT_KEYS"] = False

log = logging.getLogger(__name__)

# -------------------------------------------------------------------- helpers
REQUIRED_FIELDS = (
    "geohash",
    "timestamp",
    "day",
    "RoadType",
    "Weather",
    "Temperature",
    "NumberofLanes",
    "LargeVehicles",
    "Landmarks",
)

ERROR_MODELS_NOT_LOADED = {
    "error": "Models not loaded. Run save_models.py first.",
}

# Load artifacts once at startup; failures are logged but do not crash the
# app — the predict routes return a clean error instead.
ARTIFACTS: dict | None = None
try:
    ARTIFACTS = load_artifacts()
    log.info(
        "Artifacts loaded: method=%s, n_features=%d, fold_models=%s",
        ARTIFACTS["ensemble_meta"].get("method"),
        len(ARTIFACTS["ensemble_meta"]["feature_cols"]),
        {k: len(v) for k, v in ARTIFACTS["fold_models"].items()},
    )
except Exception as exc:
    log.exception("Failed to load inference artifacts: %s", exc)
    ARTIFACTS = None


def _predict_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Score a single record; return JSON-ready dict."""
    if ARTIFACTS is None:
        return ERROR_MODELS_NOT_LOADED, 503
    try:
        value = float(predict_single(payload, ARTIFACTS))
    except Exception as exc:
        log.exception("predict_single failed: %s", exc)
        return {"error": f"Prediction failed: {exc}"}, 500
    return {
        "demand": round(value, 6),
        "demand_pct": round(value * 100, 2),
    }, 200


# -------------------------------------------------------------------- routes
@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health() -> tuple[dict, int]:
    """Lightweight readiness probe."""
    if ARTIFACTS is None:
        return {"status": "unavailable", "reason": "artifacts not loaded"}, 503
    return {
        "status": "ok",
        "method": ARTIFACTS["ensemble_meta"].get("method"),
        "n_features": len(ARTIFACTS["ensemble_meta"]["feature_cols"]),
    }, 200


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be JSON object"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    body, status = _predict_response(payload)
    return jsonify(body), status


@app.route("/predict_batch", methods=["POST"])
def predict_batch_route():
    if ARTIFACTS is None:
        return jsonify(ERROR_MODELS_NOT_LOADED), 503
    if "file" not in request.files:
        return jsonify({"error": "No file part in request (expected field 'file')"}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected"}), 400

    try:
        df = pd.read_csv(io.StringIO(file.read().decode("utf-8")))
    except UnicodeDecodeError:
        return jsonify({"error": "File must be UTF-8 encoded CSV"}), 400
    except Exception as exc:  # parser errors, empty files, etc.
        return jsonify({"error": f"Failed to parse CSV: {exc}"}), 400

    if df.empty:
        return jsonify({"error": "CSV is empty"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in df.columns]
    if missing:
        return jsonify(
            {"error": f"CSV missing required columns: {', '.join(missing)}"}
        ), 400

    try:
        df["demand"] = predict_batch(df, ARTIFACTS).round(6)
    except Exception as exc:
        log.exception("predict_batch failed: %s", exc)
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    out_buf = io.StringIO()
    df.to_csv(out_buf, index=False)
    out_buf.seek(0)

    return send_file(
        io.BytesIO(out_buf.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="predictions.csv",
    )


# -------------------------------------------------------------------- main
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app.run(debug=True, host="127.0.0.1", port=5000)