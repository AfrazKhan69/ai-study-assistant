import json
import os
import re
import shutil

import faiss
import numpy as np
import ollama

from pdf_chunks import build_chunks

EMBED_MODEL = "nomic-embed-text"
DATA_DIR = "data"

_cache = {}  # doc_name -> (file modified time, index, chunks)


def safe_name(filename):
    """Turn 'My Notes (1).pdf' into 'My_Notes_1' so it works as a folder name."""
    name = os.path.splitext(filename)[0]
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip("_") or "document"


def doc_dir(doc_name):
    return os.path.join(DATA_DIR, doc_name)


def list_documents():
    """Names of all processed documents."""
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.exists(os.path.join(DATA_DIR, d, "index.faiss"))
    )


def delete_document(doc_name):
    shutil.rmtree(doc_dir(doc_name), ignore_errors=True)
    _cache.pop(doc_name, None)


def embed_texts(texts, prefix, progress=None, batch_size=32):
    """Turn a list of texts into a normalized float32 matrix."""
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[i:i + batch_size]]
        result = ollama.embed(model=EMBED_MODEL, input=batch)
        vectors.extend(result["embeddings"])
        if progress:
            progress(min((i + batch_size) / len(texts), 1.0))
    matrix = np.array(vectors, dtype="float32")
    faiss.normalize_L2(matrix)  # makes dot product = cosine similarity
    return matrix


def build_index(pdf_path, doc_name, progress=None):
    chunks = build_chunks(pdf_path)
    if not chunks:
        raise ValueError("No text found in this PDF (it may be scanned).")

    texts = [c["text"] for c in chunks]
    matrix = embed_texts(texts, prefix="search_document: ", progress=progress)

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    folder = doc_dir(doc_name)
    os.makedirs(folder, exist_ok=True)
    faiss.write_index(index, os.path.join(folder, "index.faiss"))
    with open(os.path.join(folder, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    _cache.pop(doc_name, None)


def load_doc(doc_name):
    """Load a document's index and chunks, using the cache when possible."""
    folder = doc_dir(doc_name)
    index_path = os.path.join(folder, "index.faiss")
    mtime = os.path.getmtime(index_path)

    cached = _cache.get(doc_name)
    if cached and cached[0] == mtime:
        return cached[1], cached[2]

    index = faiss.read_index(index_path)
    with open(os.path.join(folder, "chunks.json"), encoding="utf-8") as f:
        chunks = json.load(f)
    _cache[doc_name] = (mtime, index, chunks)
    return index, chunks


def search(question, doc_names, top_k=4):
    """Search the chosen documents and return the best chunks overall."""
    if not doc_names:
        return []

    q_vec = embed_texts([question], prefix="search_query: ")

    results = []
    for name in doc_names:
        index, chunks = load_doc(name)
        scores, ids = index.search(q_vec, min(top_k, index.ntotal))
        for score, i in zip(scores[0], ids[0]):
            if i < 0:  # FAISS returns -1 when it has fewer results than asked
                continue
            results.append({"score": float(score), "doc": name, **chunks[i]})

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]