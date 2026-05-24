import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.hybrid_retriever import HybridRetriever
from src.reranker import BGEReranker
from src.prompt_template import build_rag_prompt


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

    reranker = BGEReranker(device="cpu")
    chunks = reranker.rerank(
        query=args.question,
        candidates=candidates,
        top_k=args.top_k
    )

    prompt = build_rag_prompt(args.question, chunks)

    print(prompt)


if __name__ == "__main__":
    main()
