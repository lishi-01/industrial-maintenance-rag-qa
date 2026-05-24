from typing import List, Dict
from FlagEmbedding import FlagReranker


class BGEReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu"
    ):
        self.reranker = FlagReranker(
            model_name,
            use_fp16=False,
            devices=[device]
        )

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        if not candidates:
            return []

        pairs = [
            [query, item["text"]]
            for item in candidates
        ]

        scores = self.reranker.compute_score(pairs)

        if isinstance(scores, float):
            scores = [scores]

        reranked = []

        for item, score in zip(candidates, scores):
            new_item = dict(item)
            new_item["rerank_score"] = float(score)
            reranked.append(new_item)

        reranked = sorted(
            reranked,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]
