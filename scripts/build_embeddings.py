import argparse
import json
import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", required=True, help="Path to processed_docs.json")
    parser.add_argument("--index_path", required=True, help="Where to store FAISS index")
    args = parser.parse_args()

    # Load processed docs
    with open(args.docs, "r", encoding="utf-8") as f:
        docs = json.load(f)

    # Extract text chunks
    if isinstance(docs[0], str):
        texts = docs
    else:
        texts = [d["text"] for d in docs]

    print(f"Loaded {len(texts)} text chunks")

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generating embeddings...")
    embeddings = model.encode(texts)

    # Convert to FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    embeddings_np = np.array(embeddings).astype("float32")
    index.add(embeddings_np)

    # Ensure folder exists
    os.makedirs(args.index_path, exist_ok=True)

    # Save FAISS index in correct format for your Streamlit app
    faiss.write_index(index, os.path.join(args.index_path, "faiss.index"))

    # Save IDs
    ids = list(range(len(texts)))
    with open(os.path.join(args.index_path, "ids.pkl"), "wb") as f:
        pickle.dump(ids, f)

    # Save chunks
    with open(os.path.join(args.index_path, "chunks.pkl"), "wb") as f:
        pickle.dump(texts, f)

    print(f"FAISS index saved to {args.index_path}")
    print("Saved: faiss.index, ids.pkl, chunks.pkl")


if __name__ == "__main__":
    main()
