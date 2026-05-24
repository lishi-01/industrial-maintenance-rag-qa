import argparse
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection


COLLECTION_NAME = "industrial_maintenance_chunks"
EMBEDDING_MODEL = "BAAI/bge-m3"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    connections.connect(alias="default", host="localhost", port="19530")

    collection = Collection(COLLECTION_NAME)
    collection.load()

    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    query_embedding = model.encode(
        [args.question],
        normalize_embeddings=True
    )

    search_params = {
        "metric_type": "COSINE",
        "params": {
            "ef": 64
        }
    }

    results = collection.search(
        data=query_embedding.tolist(),
        anns_field="embedding",
        param=search_params,
        limit=args.top_k,
        output_fields=[
            "chunk_id",
            "source_file",
            "page_start",
            "page_end",
            "text",
            "language",
            "equipment",
            "fault_type"
        ]
    )

    print(f"\nQuestion: {args.question}")
    print(f"Top-{args.top_k} retrieved chunks:\n")

    for rank, hit in enumerate(results[0], start=1):
        entity = hit.entity
        text = entity.get("text", "")
        preview = text.replace("\n", " ")[:500]

        print("=" * 100)
        print(f"Rank: {rank}")
        print(f"Score: {hit.score:.4f}")
        print(f"Chunk ID: {entity.get('chunk_id')}")
        print(f"Source: {entity.get('source_file')}, page {entity.get('page_start')}")
        print(f"Text: {preview}")


if __name__ == "__main__":
    main()
