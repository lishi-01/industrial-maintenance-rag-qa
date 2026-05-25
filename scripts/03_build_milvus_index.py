import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_rag_config, get_config_value
from src.embedder import BGEEmbedder
from src.milvus_store import MilvusStore


def load_chunks(path: Path):
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def main():
    config = load_rag_config()

    chunks_path = Path(get_config_value(config, "document.chunks_path", "data/chunks/chunks.jsonl"))

    embedding_model = get_config_value(config, "embedding.model_name", "BAAI/bge-m3")
    embedding_device = get_config_value(config, "embedding.device", "cpu")
    embedding_batch_size = int(get_config_value(config, "embedding.batch_size", 16))

    milvus_host = get_config_value(config, "milvus.host", "localhost")
    milvus_port = str(get_config_value(config, "milvus.port", "19530"))
    collection_name = get_config_value(config, "milvus.collection_name", "industrial_maintenance_chunks")

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    chunks = load_chunks(chunks_path)
    print(f"Loaded chunks: {len(chunks)}")
    print(f"Embedding model: {embedding_model}")
    print(f"Embedding device: {embedding_device}")
    print(f"Milvus collection: {collection_name}")

    texts = [c["text"] for c in chunks]

    embedder = BGEEmbedder(
        model_name=embedding_model,
        device=embedding_device
    )

    embeddings = embedder.encode(
        texts,
        batch_size=embedding_batch_size
    )

    print(f"Embedding shape: {embeddings.shape}")

    store = MilvusStore(
        host=milvus_host,
        port=milvus_port,
        collection_name=collection_name,
        dim=embeddings.shape[1],
    )

    store.recreate_collection()
    num_entities = store.insert_chunks(chunks, embeddings)

    print(f"Inserted entities: {num_entities}")
    print(f"Milvus collection: {collection_name}")


if __name__ == "__main__":
    main()
