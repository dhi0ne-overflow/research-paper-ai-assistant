import faiss
import numpy as np
import os
import pickle

def get_paper_path(paper_name):
    base = os.path.join("data", "papers", paper_name)
    os.makedirs(base, exist_ok=True)
    return base


def build_index(embeddings):
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_index(index, chunks, paper_name):
    path = get_paper_path(paper_name)

    faiss.write_index(index, os.path.join(path, "index.bin"))

    with open(os.path.join(path, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)


def load_index(paper_name):
    path = get_paper_path(paper_name)

    index_path = os.path.join(path, "index.bin")
    chunks_path = os.path.join(path, "chunks.pkl")

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        return None, None

    try:
        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
    except:
        # corrupted → reset
        return None, None

    return index, chunks


def list_papers():
    base = os.path.join("data", "papers")
    if not os.path.exists(base):
        return []
    return sorted(os.listdir(base))


def retrieve_chunks(index, query_embedding, chunks, top_k=8):
    query_embedding = np.array(query_embedding).astype("float32")
    D, I = index.search(query_embedding, top_k)

    results = []
    for i in I[0]:
        if 0 <= i < len(chunks):
            results.append(chunks[i])

    return results