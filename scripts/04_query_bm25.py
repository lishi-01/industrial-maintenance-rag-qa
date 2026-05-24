import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.bm25_retriever import BM25Retriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    retriever = BM25Retriever("data/chunks/chunks.jsonl")
    results = retriever.search(args.question, top_k=args.top_k)

    print(f"\nQuestion: {args.question}")
    print(f"Top-{args.top_k} BM25 retrieved chunks:\n")

    for item in results:
        preview = item["text"].replace("\n", " ")[:500]

        print("=" * 100)
        print(f"Rank: {item['rank']}")
        print(f"BM25 Score: {item['score']:.4f}")
        print(f"Chunk ID: {item['chunk_id']}")
        print(f"Source: {item['source_file']}, page {item['page_start']}")
        print(f"Text: {preview}")


if __name__ == "__main__":
    main()
