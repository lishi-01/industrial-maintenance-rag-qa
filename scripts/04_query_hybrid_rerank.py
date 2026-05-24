import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.hybrid_retriever import HybridRetriever
from src.reranker import BGEReranker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    retriever = HybridRetriever()
    candidates = retriever.search(
        query=args.question,
        dense_top_k=20,
        bm25_top_k=20,
        final_top_k=30
    )

    reranker = BGEReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        device="cpu"
    )

    results = reranker.rerank(
        query=args.question,
        candidates=candidates,
        top_k=args.top_k
    )

    print(f"\nQuestion: {args.question}")
    print(f"Top-{args.top_k} reranked chunks:\n")

    for rank, item in enumerate(results, start=1):
        preview = item["text"].replace("\n", " ")[:600]

        print("=" * 100)
        print(f"Rank: {rank}")
        print(f"RRF Score: {item.get('rrf_score', 0):.6f}")
        print(f"Rerank Score: {item['rerank_score']:.6f}")
        print(f"Chunk ID: {item['chunk_id']}")
        print(f"Source: {item['source_file']}, page {item['page_start']}")
        print(f"Text: {preview}")


if __name__ == "__main__":
    main()
