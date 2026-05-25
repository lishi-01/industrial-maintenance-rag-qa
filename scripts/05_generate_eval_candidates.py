import json
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.hybrid_retriever import HybridRetriever
from src.reranker import BGEReranker


QUESTIONS_PATH = Path("data/eval/seed_questions.jsonl")
OUT_PATH = Path("data/eval/retrieval_eval_candidates.jsonl")


def main():
    retriever = HybridRetriever(device="cpu")
    reranker = BGEReranker(device="cpu")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with QUESTIONS_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Generating eval candidates"):
            item = json.loads(line)
            question = item["question"]

            candidates = retriever.search(
                query=question,
                dense_top_k=20,
                bm25_top_k=20,
                final_top_k=30
            )

            reranked = reranker.rerank(
                query=question,
                candidates=candidates,
                top_k=10
            )

            output = {
                "question": question,
                "candidates": [
                    {
                        "rank": idx + 1,
                        "chunk_id": c.get("chunk_id"),
                        "source_file": c.get("source_file"),
                        "page_start": c.get("page_start"),
                        "page_end": c.get("page_end"),
                        "rrf_score": c.get("rrf_score"),
                        "rerank_score": c.get("rerank_score"),
                        "text_preview": c.get("text", "").replace("\n", " ")[:500]
                    }
                    for idx, c in enumerate(reranked)
                ],
                "gold_sources": []
            }

            fout.write(json.dumps(output, ensure_ascii=False) + "\n")

    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
