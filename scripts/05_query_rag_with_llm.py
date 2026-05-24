import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.rag_chain import IndustrialRAGChain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    rag = IndustrialRAGChain(
        llm_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        device="cpu"
    )

    result = rag.query(
        question=args.question,
        final_top_k=args.top_k
    )

    print("\n" + "=" * 100)
    print("Question:")
    print(result["question"])

    print("\n" + "=" * 100)
    print("Answer:")
    print(result["answer"])

    print("\n" + "=" * 100)
    print("Sources:")
    for idx, source in enumerate(result["sources"], start=1):
        print("-" * 100)
        print(f"[{idx}] {source['source_file']}, page {source['page_start']}")
        print(f"chunk_id: {source['chunk_id']}")
        print(f"rrf_score: {source['rrf_score']}")
        print(f"rerank_score: {source['rerank_score']}")
        print(f"preview: {source['text'].replace(chr(10), ' ')[:300]}")


if __name__ == "__main__":
    main()
