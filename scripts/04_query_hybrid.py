import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.hybrid_retriever import HybridRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    retriever = HybridRetriever()
    results = retriever.search(
        query=args.question,
        dense_top_k=20,
        bm25_top_k=20,
        final_top_k=args.top_k
    )

    print(f"\nQuestion: {args.question}")
    print(f"Top-{args.top_k} Hybrid retrieved chunks:\n")

    for rank, item in enumerate(results, start=1):
        preview = item["text"].replace("\n", " ")[:500]

        print("=" * 100)
        print(f"Rank: {rank}")
        print(f"RRF Score: {item['rrf_score']:.6f}")
        print(f"Chunk ID: {item['chunk_id']}")
        print(f"Source: {item['source_file']}, page {item['page_start']}")
        print(f"Text: {preview}")


if __name__ == "__main__":
    main()
