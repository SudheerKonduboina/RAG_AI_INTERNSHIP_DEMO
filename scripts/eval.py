# scripts/eval.py
import time
import json
from sentence_transformers import SentenceTransformer
import faiss
import pickle

INDEX_PATH = 'data/faiss_index'
with open('data/processed_docs.json','r',encoding='utf-8') as f:
    docs = json.load(f)

with open(f"{INDEX_PATH}/ids.pkl", 'rb') as f:
    ids = pickle.load(f)
index = faiss.read_index(f"{INDEX_PATH}/index.faiss")
model = SentenceTransformer('all-MiniLM-L6-v2')

queries = [
    {'q':'How do I reset my password?', 'expected_ids': ['row0_chunk0']},
]

for q in queries:
    t0 = time.time()
    emb = model.encode([q['q']], convert_to_numpy=True)
    import numpy as np
    faiss.normalize_L2(emb)
    D,I = index.search(emb, 5)
    t = time.time()-t0
    retrieved = [ids[i] for i in I[0]]
    hit = any(e in retrieved for e in q['expected_ids'])
    print(q['q'], 'retrieved:', retrieved, 'hit:', hit, 'latency:', t)
