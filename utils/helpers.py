# utils/helpers.py
import re
from typing import List

def clean_text(text: str) -> str:
    text = text.replace('\r', ' ').replace('\n', ' ')
    text = re.sub(r"\s+", ' ', text)
    text = text.strip()
    return text

def chunk_text(text: str, chunk_size:int=100, overlap:int=20) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap
    return chunks
