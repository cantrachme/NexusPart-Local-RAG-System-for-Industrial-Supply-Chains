import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import faiss

# -----------------------------
# Config
# -----------------------------

CSV_PATH = "clean_parts.csv"
FAISS_PATH = "faiss.index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Local MLflow (no server needed yet)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("NexusPart-SemanticSearch")

# -----------------------------
# Load data
# -----------------------------

print("Loading CSV...")
df = pd.read_csv(CSV_PATH)

texts = df["combined_text"].astype(str).tolist()

# -----------------------------
# Embeddings
# -----------------------------

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("Creating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# -----------------------------
# FAISS
# -----------------------------

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, FAISS_PATH)

print("FAISS vectors:", index.ntotal)

# -----------------------------
# MLflow tracking
# -----------------------------

with mlflow.start_run():
    mlflow.log_param("model", MODEL_NAME)
    mlflow.log_param("rows", len(df))
    mlflow.log_param("dimension", dim)

    mlflow.log_metric("vector_count", index.ntotal)

    # Save artifacts
    joblib.dump(df, "parts_df.pkl")
    mlflow.log_artifact("parts_df.pkl")
    mlflow.log_artifact(FAISS_PATH)

    print("Run logged to MLflow")

print("Done.")
