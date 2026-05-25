import json
import re
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.rag_chain import IndustrialRAGChain


EVAL_PATH = Path("data/eval/qa_eval_set.jsonl")
OUT_PATH = Path("data/eval/answer_eval_results.jsonl")
REPORT_PATH = Path("docs/answer_eval_report.md")


def tokenize(text: str):
    text = text.lower()
    return re.findall(r"[a-zA-Z]+|[\u4e00-\u9fff]+|\d+", text)


def point_covered(answer: str, point: str):
    answer_tokens = set(tokenize(answer))
    point_tokens = tokenize(point)

    if not point_tokens:
        return False

    overlap = sum(1 for t in point_tokens if t in answer_tokens)
    ratio = overlap / len(point_tokens)

    return ratio >= 0.35 or overlap >= 3


def compute_answer_coverage(answer: str, gold_points):
    if not gold_points:
        return 0.0, []

    covered = []
    for point in gold_points:
        covered.append(point_covered(answer, point))

    return sum(covered) / len(gold_points), covered


def check_citation_presence(answer: str, sources):
    if not sources:
        return False

    for src in sources:
        source_file = src.get("source_file", "")
        page = str(src.get("page_start", ""))

        if source_file and source_file in answer:
            return True

        if page and (f"第 {page} 页" in answer or f"page {page}" in answer.lower()):
            return True

    return False


def main():
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Eval file not found: {EVAL_PATH}")

    eval_items = []
    with EVAL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eval_items.append(json.loads(line))

    rag = IndustrialRAGChain()

    results = []

    for item in tqdm(eval_items, desc="Evaluating answer generation"):
        question = item["question"]
        gold_points = item["gold_answer_points"]
        must_cite = item.get("must_cite", True)

        rag_result = rag.query(
            question=question,
            final_top_k=5
        )

        answer = rag_result["answer"]
        sources = rag_result["sources"]

        coverage, covered_flags = compute_answer_coverage(answer, gold_points)
        citation_present = check_citation_presence(answer, sources)

        hallucination_flag = False
        if coverage < 0.34:
            hallucination_flag = True
        if must_cite and not citation_present:
            hallucination_flag = True

        results.append({
            "question": question,
            "answer": answer,
            "gold_answer_points": gold_points,
            "covered_flags": covered_flags,
            "answer_coverage": coverage,
            "citation_present": citation_present,
            "hallucination_flag": hallucination_flag,
            "sources": sources
        })

    avg_coverage = sum(r["answer_coverage"] for r in results) / len(results)
    citation_rate = sum(1 for r in results if r["citation_present"]) / len(results)
    hallucination_rate = sum(1 for r in results if r["hallucination_flag"]) / len(results)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("# Answer Generation Evaluation Report\n\n")
        f.write(f"- Number of questions: {len(results)}\n")
        f.write(f"- Answer Coverage: {avg_coverage:.4f}\n")
        f.write(f"- Citation Presence Rate: {citation_rate:.4f}\n")
        f.write(f"- Hallucination Flag Rate: {hallucination_rate:.4f}\n\n")

    print("Answer evaluation completed.")
    print(f"Questions: {len(results)}")
    print(f"Answer Coverage: {avg_coverage:.4f}")
    print(f"Citation Presence Rate: {citation_rate:.4f}")
    print(f"Hallucination Flag Rate: {hallucination_rate:.4f}")
    print(f"Saved results to: {OUT_PATH}")
    print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
