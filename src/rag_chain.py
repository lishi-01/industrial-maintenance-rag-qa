from typing import Dict, List

from src.hybrid_retriever import HybridRetriever
from src.reranker import BGEReranker
from src.prompt_template import build_rag_prompt
from src.llm_client import LocalQwenClient


class IndustrialRAGChain:
    def __init__(
        self,
        llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cpu",
    ):
        self.retriever = HybridRetriever(device=device)
        self.reranker = BGEReranker(device=device)
        self.llm = LocalQwenClient(
            model_name=llm_model_name,
            device="auto"
        )

    def query(
        self,
        question: str,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        candidate_top_k: int = 30,
        final_top_k: int = 5,
    ) -> Dict:
        candidates = self.retriever.search(
            query=question,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            final_top_k=candidate_top_k
        )

        evidence_chunks = self.reranker.rerank(
            query=question,
            candidates=candidates,
            top_k=final_top_k
        )

        prompt = build_rag_prompt(
            question=question,
            chunks=evidence_chunks
        )

        answer = self.llm.generate(
            prompt=prompt,
            temperature=0.1,
            top_p=0.9,
            max_new_tokens=1024
        )

        sources = []
        for item in evidence_chunks:
            sources.append({
                "chunk_id": item.get("chunk_id"),
                "source_file": item.get("source_file"),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "rrf_score": item.get("rrf_score"),
                "rerank_score": item.get("rerank_score"),
                "text": item.get("text", "")[:500]
            })

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "evidence_chunks": evidence_chunks,
            "prompt": prompt
        }
