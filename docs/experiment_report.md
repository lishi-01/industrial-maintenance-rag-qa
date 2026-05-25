# Experiment Report

## 1. Retrieval Evaluation

We constructed a retrieval evaluation set with 10 industrial bearing maintenance questions.  
Each question was manually annotated with relevant gold sources based on retrieved document chunks.

### Metrics

| Metric | Score |
|---|---:|
| Recall@5 | 1.0000 |
| Recall@10 | 1.0000 |
| MRR | 0.9250 |
| nDCG@5 | 0.8728 |

### Interpretation

The retrieval module achieves perfect Recall@5 and Recall@10 on the current evaluation set, indicating that the correct evidence sources can be retrieved within the top-ranked candidates. The MRR score of 0.9250 shows that most correct evidence chunks appear at very high ranks. The nDCG@5 score of 0.8728 further indicates that the reranked results are generally well ordered.

## 2. Current Pipeline

The current MVP pipeline includes:

- PDF parsing with page-level metadata
- Sliding-window chunking
- BAAI/bge-m3 dense embedding
- Milvus vector storage
- BM25 sparse retrieval
- Dense + BM25 hybrid retrieval with RRF fusion
- BAAI/bge-reranker-v2-m3 reranking
- Local Qwen2.5 generation
- FastAPI backend
- Gradio demo interface
- Source citation with file name, page number, chunk ID, and reranker score

