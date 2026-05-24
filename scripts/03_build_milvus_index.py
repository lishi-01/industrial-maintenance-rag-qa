import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.embedder import BGEEmbedder
from src.milvus_store import MilvusStore


CHUNKS_PATH = Path("data/chunks/chunks.jsonl")
COLLECTION_NAME = "industrial_maintenance_chunks"


def load_chunks(path: Path):
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")

    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded chunks: {len(chunks)}")

    texts = [c["text"] for c in chunks]

    embedder = BGEEmbedder(
        model_name="BAAI/bge-m3",
        device="cpu"
    )

    embeddings = embedder.encode(
        texts,
        batch_size=16,
    )

    print(f"Embedding shape: {embeddings.shape}")

    store = MilvusStore(
        host="localhost",
        port="19530",
        collection_name=COLLECTION_NAME,
        dim=embeddings.shape[1],
    )

    store.recreate_collection()
    num_entities = store.insert_chunks(chunks, embeddings)

    print(f"Inserted entities: {num_entities}")
    print(f"Milvus collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
