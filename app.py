import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import subprocess

# =========================
# Page Config
# =========================

st.set_page_config(page_title="NexusPart AI", page_icon="🔧", layout="wide")

# =========================
# Load Resources (Cached)
# =========================


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_index():
    return faiss.read_index("faiss.index")


@st.cache_data
def load_data():
    return pd.read_csv("clean_parts.csv")


model = load_model()
index = load_index()
df = load_data()

# =========================
# Ollama Helper
# =========================


def ask_phi3(prompt):
    result = subprocess.run(
        ["ollama", "run", "phi3"], input=prompt.encode("utf-8"), capture_output=True
    )
    return result.stdout.decode("utf-8", errors="ignore")


# =========================
# Header
# =========================

st.markdown(
    """
<h1 style='text-align:center'>🔧 NexusPart AI</h1>
<p style='text-align:center'>Semantic Industrial Parts Recommendation Engine</p>
<hr>
""",
    unsafe_allow_html=True,
)

# =========================
# Session State
# =========================

if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# Sidebar
# =========================

with st.sidebar:
    st.header("⚙️ Controls")

    top_k = st.slider("Number of Results", 1, 10, 5)
    show_ai = st.checkbox("Show AI Explanation", True)

    st.divider()

    st.subheader("🕘 Search History")
    for q in st.session_state.history[-5:]:
        st.write(q)

    st.divider()
    st.caption("NexusPart AI • Local RAG System")

# =========================
# Main Input
# =========================

query = st.text_area(
    "Enter part description:",
    height=120,
    placeholder="Example: Indicator Red Fast Movement 1.6A 250V Holder Plastic 5x20mm...",
)

# =========================
# Search Logic
# =========================

if query:
    st.session_state.history.append(query)

    with st.spinner("🔎 Searching similar parts..."):
        q_emb = model.encode([query])
        D, I = index.search(np.array(q_emb).astype("float32"), top_k)

        results = df.iloc[I[0]].copy()
        results["distance"] = D[0]

    # =========================
    # Metrics
    # =========================

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Parts", len(df))
    col2.metric("Vector Count", index.ntotal)
    col3.metric("Results Returned", len(results))

    st.divider()

    # =========================
    # Results Cards
    # =========================

    st.subheader("🔍 Similar Parts")

    for _, row in results.iterrows():
        st.markdown(
            f"""
        <div style="
            border-radius:12px;
            padding:15px;
            margin-bottom:12px;
            background-color:#1e1e1e;
            border:1px solid #333;">
        <h4>{row["ID"]}</h4>
        <p>{row["DESCRIPTION"]}</p>
        <small>Distance: {row["distance"]:.3f}</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

        with st.expander("📋 Technical Details"):
            st.json(row.to_dict())


# =========================
# Footer
# =========================

st.markdown(
    """
<hr>
<p style='text-align:center;color:gray'>
Built with FAISS • SentenceTransformers • Ollama • Streamlit
</p>
""",
    unsafe_allow_html=True,
)
