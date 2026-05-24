import json
from pathlib import Path
from typing import List, Dict

from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection

from src.bm25_retriever import BM25Retriever


COLLECTION_NAME = "industrial_maintenance_chunks"
EMBEDDING_MODEL = "BAAI/bge-m3"


def reciprocal_rank_fusion(
    dense_results: List[Dict],
    bm25_results: List[Dict],
    k: int = 60,
    top_k: int = 10
) -> List[Dict]:
    fused = {}

    for rank, item in enumerate(dense_results, start=1):
        chunk_id = item["chunk_id"]
        if chunk_id not in fused:
            fused[chunk_id] = item
            fused[chunk_id]["rrf_score"] = 0.0
        fused[chunk_id]["rrf_score"] += 1.0 / (k + rank)

    for rank, item in enumerate(bm25_results, start=1):
        chunk_id = item["chunk_id"]
        if chunk_id not in fused:
            fused[chunk_id] = item
            fused[chunk_id]["rrf_score"] = 0.0
        fused[chunk_id]["rrf_score"] += 1.0 / (k + rank)

    results = sorted(
        fused.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return results[:top_k]


class HybridRetriever:
    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        chunks_path: str = "data/chunks/chunks.jsonl",
        embedding_model: str = EMBEDDING_MODEL,
        device: str = "cpu"
    ):
        self.collection_name = collection_name

        connections.connect(alias="default", host="localhost", port="19530")
        self.collection = Collection(collection_name)
        self.collection.load()

        self.embedder = SentenceTransformer(embedding_model, device=device)
        self.bm25 = BM25Retriever(chunks_path)

    def dense_search(self, query: str, top_k: int = 20) -> List[Dict]:
        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True
        )

        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 64}
        }

        results = self.collection.search(
            data=query_embedding.tolist(),
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=[
                "chunk_id",
                "doc_id",
                "source_file",
                "page_start",
                "page_end",
                "section",
                "text",
                "language",
                "equipment",
                "fault_type"
            ]
        )

        dense_results = []

        for rank, hit in enumerate(results[0], start=1):
            entity = hit.entity
            dense_results.append({
                "chunk_id": entity.get("chunk_id"),
                "doc_id": entity.get("doc_id"),
                "source_file": entity.get("source_file"),
                "page_start": entity.get("page_start"),
                "page_end": entity.get("page_end"),
                "section": entity.get("section"),
                "text": entity.get("text"),
                "language": entity.get("language"),
                "metadata": {
                    "equipment": entity.get("equipment"),
                    "fault_type": entity.get("fault_type")
                },
                "dense_score": float(hit.score),
                "dense_rank": rank
            })

        return dense_results

    def search(
        self,
        query: str,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        final_top_k: int = 10
    ) -> List[Dict]:
        dense_results = self.dense_search(query, top_k=dense_top_k)
        bm25_results = self.bm25.search(query, top_k=bm25_top_k)

        return reciprocal_rank_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            k=60,
            top_k=final_top_k
        )
