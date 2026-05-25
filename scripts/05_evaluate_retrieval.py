import json
import math
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.hybrid_retriever import HybridRetriever
from src.reranker import BGEReranker


EVAL_PATH = Path("data/eval/retrieval_eval_set.jsonl")
OUT_PATH = Path("data/eval/retrieval_eval_results.jsonl")
REPORT_PATH = Path("docs/retrieval_eval_report.md")


def load_eval_set(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def normalize_source(source_file: str):
    return source_file.strip().lower()


def is_hit(candidate, gold_sources):
    cand_file = normalize_source(candidate.get("source_file", ""))
    cand_page = int(candidate.get("page_start", -1))

    for gold in gold_sources:
        gold_file = normalize_source(gold["source_file"])
        gold_page = int(gold["page"])

        if cand_file == gold_file and cand_page == gold_page:
            return True

    return False


def recall_at_k(candidates, gold_sources, k):
    top_k = candidates[:k]
    return 1.0 if any(is_hit(c, gold_sources) for c in top_k) else 0.0


def reciprocal_rank(candidates, gold_sources):
    for idx, c in enumerate(candidates, start=1):
        if is_hit(c, gold_sources):
            return 1.0 / idx
    return 0.0


def ndcg_at_k(candidates, gold_sources, k):
    dcg = 0.0

    for idx, c in enumerate(candidates[:k], start=1):
        rel = 1.0 if is_hit(c, gold_sources) else 0.0
        if rel > 0:
            dcg += rel / math.log2(idx + 1)

    ideal_hits = min(len(gold_sources), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def main():
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Eval file not found: {EVAL_PATH}")

    eval_set = load_eval_set(EVAL_PATH)

    retriever = HybridRetriever(device="cpu")
    reranker = BGEReranker(device="cpu")

    results = []

    for item in tqdm(eval_set, desc="Evaluating retrieval"):
        question = item["question"]
        gold_sources = item["gold_sources"]

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

        r5 = recall_at_k(reranked, gold_sources, 5)
        r10 = recall_at_k(reranked, gold_sources, 10)
        mrr = reciprocal_rank(reranked, gold_sources)
        ndcg5 = ndcg_at_k(reranked, gold_sources, 5)

        result = {
            "question": question,
            "gold_sources": gold_sources,
            "recall@5": r5,
            "recall@10": r10,
            "mrr": mrr,
            "ndcg@5": ndcg5,
            "top_results": [
                {
                    "rank": idx + 1,
                    "source_file": c.get("source_file"),
                    "page_start": c.get("page_start"),
                    "chunk_id": c.get("chunk_id"),
                    "rrf_score": c.get("rrf_score"),
                    "rerank_score": c.get("rerank_score"),
                    "hit": is_hit(c, gold_sources),
                    "text_preview": c.get("text", "").replace("\n", " ")[:300]
                }
                for idx, c in enumerate(reranked)
            ]
        }

        results.append(result)

    avg_r5 = sum(x["recall@5"] for x in results) / len(results)
    avg_r10 = sum(x["recall@10"] for x in results) / len(results)
    avg_mrr = sum(x["mrr"] for x in results) / len(results)
    avg_ndcg5 = sum(x["ndcg@5"] for x in results) / len(results)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("# Retrieval Evaluation Report\n\n")
        f.write(f"- Number of questions: {len(results)}\n")
        f.write(f"- Recall@5: {avg_r5:.4f}\n")
        f.write(f"- Recall@10: {avg_r10:.4f}\n")
        f.write(f"- MRR: {avg_mrr:.4f}\n")
        f.write(f"- nDCG@5: {avg_ndcg5:.4f}\n\n")

    print("Retrieval evaluation completed.")
    print(f"Questions: {len(results)}")
    print(f"Recall@5: {avg_r5:.4f}")
    print(f"Recall@10: {avg_r10:.4f}")
    print(f"MRR: {avg_mrr:.4f}")
    print(f"nDCG@5: {avg_ndcg5:.4f}")
    print(f"Saved results to: {OUT_PATH}")
    print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
