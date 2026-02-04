# NexusPart-Local-RAG-System-for-Industrial-Supply-Chains
•  RAG •  Local LLM deployment •  Semantic search &amp; vector databases •  MLOps &amp; system design •  Prompt engineering for reasoning •  Streamlit-based AI dashboards

# 🔧 NexusPart  
### Local Retrieval-Augmented Generation System for Industrial Parts Intelligence

NexusPart is a fully local Retrieval-Augmented Generation (RAG) system designed for semantic industrial part matching and substitution. The platform combines dense vector retrieval with local large language model reasoning to deliver explainable recommendations over proprietary parts catalogs.

The system is architected to operate entirely on-premise, enabling private inference, secure data handling, and low-latency retrieval without external APIs.

---

## Overview

Traditional keyword search fails in industrial catalogs due to inconsistent naming, vendor-specific terminology, and unstructured descriptions.

NexusPart solves this by:

- Converting part descriptions into dense semantic embeddings  
- Performing similarity search using FAISS  
- Augmenting retrieved context into structured prompts  
- Generating human-readable technical explanations using a local LLM (Phi-3 via Ollama)  

The result is an end-to-end semantic reasoning pipeline for industrial supply chain intelligence.

---

## System Architecture

User Query  
→ SentenceTransformer Embedding  
→ FAISS Vector Search (Top-K)  
→ Context Assembly  
→ Phi-3 Local Inference (Ollama)  
→ Explainable Recommendations  
→ Streamlit Interface  

This follows a classic RAG architecture with fully local execution.

---

## Core Features

- Semantic similarity search over industrial parts  
- FAISS vector indexing (persistent storage)  
- Transformer-based embeddings (all-MiniLM-L6-v2)  
- Local LLM reasoning via Ollama (Phi-3)  
- Explainable substitution logic  
- MLflow experiment tracking  
- Interactive Streamlit frontend  
- Zero cloud dependencies  

---

## Technical Stack

- Python  
- Pandas / NumPy  
- SentenceTransformers  
- FAISS  
- Ollama (Phi-3)  
- MLflow  
- Streamlit  

---

## Pipeline

### 1. Data Processing

Raw parts data is cleaned, normalized, and consolidated into a unified semantic description field.

Output: `clean_parts.csv`

---

### 2. Embedding Generation

Each part description is embedded using `all-MiniLM-L6-v2`.

Final embedding matrix:

- Shape: (663, 384)

Embeddings are stored in FAISS for high-performance similarity search.

Output: `faiss.index`

---

### 3. Vector Retrieval

Incoming queries are embedded and compared against the FAISS index using cosine similarity. Top-K closest parts are retrieved.

---

### 4. Retrieval-Augmented Generation

Retrieved parts are injected into a structured prompt and passed to Phi-3 (running locally via Ollama) to generate technical substitution explanations.

---

### 5. MLOps (MLflow)

MLflow is used to track:

- Embedding models  
- FAISS index versions  
- Dataset lineage  
- Retrieval experiments  

This enables reproducibility and iteration across retrieval pipelines.

---

### 6. Application Layer

A Streamlit dashboard provides:

- Natural language search  
- Ranked semantic matches  
- Technical metadata inspection  
- Retrieval distance visualization  
- AI-generated explanations  
- Query history  

--- 

## Repository Structure
~~~
NexusPart/
├── Dataset/
│ └── Parts.csv
├── Notebooks/
│ ├── Phase2_EDA_and_Cleaning.ipynb
│ └── Phase3_Embeddings_and_FAISS.ipynb
├── clean_parts.csv
├── faiss.index
├── mlflow_pipeline.py
├── app.py
├── requirements.txt
└── README.md
~~~

---


---

## Design Philosophy

- Fully local inference  
- Deterministic retrieval  
- Explainable recommendations  
- Production-style architecture  
- Enterprise data privacy  

This mirrors real industrial environments where cloud-based LLMs are often not permitted.

---

## Future Work

- Cross-encoder reranking  
- Hybrid lexical + vector retrieval  
- GPU acceleration  
- Dockerized deployment  
- Inventory system integration  

---









