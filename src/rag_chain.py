from typing import Dict

from src.config import load_rag_config, get_config_value
from src.hybrid_retriever import HybridRetriever
from src.reranker import BGEReranker
from src.prompt_template import build_rag_prompt
from src.llm_client import LocalQwenClient


class IndustrialRAGChain:
    def __init__(self, config_path: str = "configs/rag_config.yaml"):
        self.config = load_rag_config()

        collection_name = get_config_value(
            self.config,
            "milvus.collection_name",
            "industrial_maintenance_chunks"
        )

        chunks_path = get_config_value(
            self.config,
            "document.chunks_path",
            "data/chunks/chunks.jsonl"
        )

        embedding_model = get_config_value(
            self.config,
            "embedding.model_name",
            "BAAI/bge-m3"
        )

        embedding_device = get_config_value(
            self.config,
            "embedding.device",
            "cpu"
        )

        reranker_model = get_config_value(
            self.config,
            "reranker.model_name",
            "BAAI/bge-reranker-v2-m3"
        )

        reranker_device = get_config_value(
            self.config,
            "reranker.device",
            "cpu"
        )

        llm_model = get_config_value(
            self.config,
            "llm.model_name",
            "Qwen/Qwen2.5-1.5B-Instruct"
        )

        self.retriever = HybridRetriever(
            collection_name=collection_name,
            chunks_path=chunks_path,
            embedding_model=embedding_model,
            device=embedding_device
        )

        self.reranker = BGEReranker(
            model_name=reranker_model,
            device=reranker_device
        )

        self.llm = LocalQwenClient(
            model_name=llm_model,
            device="auto"
        )

    def query(
        self,
        question: str,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
        candidate_top_k: int | None = None,
        final_top_k: int | None = None,
    ) -> Dict:
        dense_top_k = dense_top_k or int(get_config_value(self.config, "retrieval.dense_top_k", 20))
        bm25_top_k = bm25_top_k or int(get_config_value(self.config, "retrieval.bm25_top_k", 20))
        candidate_top_k = candidate_top_k or int(get_config_value(self.config, "retrieval.rerank_top_k", 30))
        final_top_k = final_top_k or int(get_config_value(self.config, "retrieval.final_top_n", 5))

        temperature = float(get_config_value(self.config, "llm.temperature", 0.1))
        top_p = float(get_config_value(self.config, "llm.top_p", 0.9))
        max_new_tokens = int(get_config_value(self.config, "llm.max_new_tokens", 1024))

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
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens
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
