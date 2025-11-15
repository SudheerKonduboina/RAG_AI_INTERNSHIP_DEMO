from dotenv import load_dotenv
load_dotenv()

# app/api_fastapi.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle, faiss
from sentence_transformers import SentenceTransformer
import os
import openai

app = FastAPI()
INDEX_PATH = 'data/faiss_index'
EMBED_MODEL = 'all-MiniLM-L6-v2'
openai.api_key = os.getenv('OPENAI_API_KEY')

if os.path.exists(f"{INDEX_PATH}/index.faiss"):
    index = faiss.read_index(f"{INDEX_PATH}/index.faiss")
    with open(f"{INDEX_PATH}/ids.pkl", 'rb') as f:
        ids = pickle.load(f)
    with open(f"{INDEX_PATH}/meta.pkl", 'rb') as f:
        meta = pickle.load(f)
else:
    index = None
    ids = []
    meta = []

embed_model = SentenceTransformer(EMBED_MODEL)

class Query(BaseModel):
    question: str

@app.post('/query')
async def query(q: Query):
    if index is None:
        return {'answer': None, 'sources': []}
    q_emb = embed_model.encode([q.question], convert_to_numpy=True)
    import numpy as np
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, 5)
    results = [ids[i] for i in I[0]]
    prompt = f"Context ids: {results}\\nQuestion: {q.question}"
    if openai.api_key:
        resp = openai.ChatCompletion.create(model='gpt-4o-mini', messages=[{'role':'user','content':prompt}])
        return {'answer': resp['choices'][0]['message']['content'], 'sources': results}
    else:
        return {'answer': None, 'sources': results}
