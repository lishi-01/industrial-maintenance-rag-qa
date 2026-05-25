
## 11. Demo Screenshot

The Gradio interface supports industrial maintenance question answering with retrieved evidence, source file names, page numbers, chunk IDs, and reranker scores.

![Gradio Demo](docs/gradio_demo_answer.png)


## 12. Retrieval Evaluation

| Metric | Score |
|---|---:|
| Recall@5 | 1.0000 |
| Recall@10 | 1.0000 |
| MRR | 0.9250 |
| nDCG@5 | 0.8728 |

The current retrieval pipeline uses BGE-M3 dense retrieval, BM25 sparse retrieval, RRF fusion, and BGE reranker. The evaluation set contains manually annotated bearing maintenance QA samples.


## 13. Answer Generation Evaluation

| Metric | Score |
|---|---:|
| Answer Coverage | 0.9333 |
| Citation Presence Rate | 1.0000 |
| Hallucination Flag Rate | 0.0000 |

The answer generation module uses retrieved and reranked evidence chunks as context. The prompt constrains the local Qwen model to cite only provided source files, page numbers, and chunk IDs.

