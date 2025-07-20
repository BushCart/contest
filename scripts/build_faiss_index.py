from pathlib import Path
import json
import numpy as np
import faiss
import pickle

EMBEDDINGS_PATH = Path("embeddings/embeddings.jsonl")
INDEX_PATH = Path("vector_store/faiss.index")
META_PATH = Path("vector_store/metadata.pkl")

vectors = []
metadata = {}
next_id = 0

with EMBEDDINGS_PATH.open("r", encoding="utf-8") as file:
    for line in file:
        chunk = json.loads(line)
        vectors.append(chunk["embedding"])
        metadata[next_id] = {
            "text": chunk["text"],
            "source": chunk.get("source"),
            "page": chunk.get("page"),
            "chunk_id": chunk.get("chunk_id"),
        }

        next_id += 1

vectors_np = np.array(vectors).astype("float32")
print(f"[✓] Загружено {len(vectors)} эмбеддингов")
index = faiss.IndexFlatL2(vectors_np.shape[1])
index.add(vectors_np)
INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

faiss.write_index(index, str(INDEX_PATH))
with META_PATH.open("wb") as f:
    pickle.dump(metadata, f)

print(f"[💾] Сохранено: {INDEX_PATH.name}, {META_PATH.name}")
