# app/streamlit_app.py
from dotenv import load_dotenv
load_dotenv()

import os
import pickle
import time
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import openai

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
INDEX_PATH = "data/faiss_index"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

# Load API key from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

st.set_page_config(page_title="RAG Chat", layout="wide")
st.title("RAG — Retrieval Augmented Chat (Internship Demo)")

# -----------------------------------------------------------
# LOAD FAISS INDEX + METADATA
# -----------------------------------------------------------
index = None
ids = []
chunks = []

index_file = os.path.join(INDEX_PATH, "faiss.index")
ids_file = os.path.join(INDEX_PATH, "ids.pkl")
chunks_file = os.path.join(INDEX_PATH, "chunks.pkl")

if os.path.exists(index_file):
    st.info("FAISS index loaded successfully.")
    index = faiss.read_index(index_file)

    if os.path.exists(ids_file) and os.path.exists(chunks_file):
        with open(ids_file, "rb") as f:
            ids = pickle.load(f)
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)
    else:
        st.warning("ids.pkl or chunks.pkl not found. Make sure embeddings were saved correctly.")
else:
    st.warning("FAISS index not found. Run the embedding script first.")

# Load embedding model
embed_model = SentenceTransformer(EMBED_MODEL)

# -----------------------------------------------------------
# RETRIEVAL FUNCTION
# -----------------------------------------------------------
def retrieve(query: str, k: int = TOP_K):
    if index is None or not chunks:
        return []

    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)

    results = []
    for i in I[0]:
        results.append({
            "id": ids[i] if ids else i,
            "text": chunks[i] if chunks else ""
        })
    return results

# -----------------------------------------------------------
# OPENAI COMPLETION FUNCTION
# -----------------------------------------------------------
def generate_with_openai(prompt: str):
    if not OPENAI_API_KEY:
        return "OpenAI API key missing. Set OPENAI_API_KEY in your .env file."

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"OpenAI Error: {str(e)}"

# -----------------------------------------------------------
# PROMPT COMPOSER
# -----------------------------------------------------------
def compose_prompt(query: str, retrieved_chunks: list):
    context = "\n\n".join([f"[Chunk {r['id']}]:\n{r['text']}" for r in retrieved_chunks])
    prompt = f"""
You are a helpful AI assistant. Use ONLY the context below to answer the question.

CONTEXT:
{context}

QUESTION:
{query}

If the answer is not available in the context, respond: "The answer is not available in the provided documents."
"""
    return prompt

# -----------------------------------------------------------
# UI — CHAT
# -----------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask a question about the dataset:")

if st.button("Ask") and query:
    t0 = time.time()
    retrieved = retrieve(query)
    retrieval_time = time.time() - t0

    prompt = compose_prompt(query, retrieved)

    t1 = time.time()
    answer = generate_with_openai(prompt)
    generation_time = time.time() - t1

    # Display results
    st.write("### Answer")
    st.write(answer)

    st.write("### Retrieved Chunks")
    for r in retrieved:
        st.write(f"**Chunk ID:** {r['id']}")
        st.write(r["text"])
        st.write("---")

    st.write(f"Retrieval Time: {retrieval_time:.3f}s")
    st.write(f"Generation Time: {generation_time:.3f}s")
