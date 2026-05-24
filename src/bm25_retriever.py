import json
import re
from pathlib import Path
from typing import List, Dict

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z]+(?:-[a-zA-Z]+)*|\d+(?:\.\d+)?|[\u4e00-\u9fff]", text)
    return tokens


class BM25Retriever:
    def __init__(self, chunks_path: str = "data/chunks/chunks.jsonl"):
        self.chunks_path = Path(chunks_path)
        self.chunks = self._load_chunks()
        self.corpus_tokens = [tokenize(chunk["text"]) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def _load_chunks(self) -> List[Dict]:
        chunks = []
        with self.chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        return chunks

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            chunk["rank"] = rank
            results.append(chunk)

        return results
