.PHONY: help parse chunk index retrieve rag api gradio eval-retrieval eval-answer test docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  make docker-up        Start Milvus standalone"
	@echo "  make docker-down      Stop Milvus standalone"
	@echo "  make parse            Parse PDFs into documents.jsonl"
	@echo "  make chunk            Build chunks.jsonl"
	@echo "  make index            Build Milvus vector index"
	@echo "  make retrieve         Test hybrid retrieval + reranker"
	@echo "  make rag              Test full RAG QA with local Qwen"
	@echo "  make api              Start FastAPI service"
	@echo "  make gradio           Start Gradio demo"
	@echo "  make eval-retrieval   Run retrieval evaluation"
	@echo "  make eval-answer      Run answer generation evaluation"
	@echo "  make test             Run unit tests"

docker-up:
	docker compose up -d

docker-down:
	docker compose down

parse:
	python scripts/01_parse_docs.py

chunk:
	python scripts/02_build_chunks.py

index:
	python scripts/03_build_milvus_index.py

retrieve:
	python scripts/04_query_hybrid_rerank.py \
		--question "bearing outer race fault BPFO spectrum" \
		--top_k 5

rag:
	python scripts/05_query_rag_with_llm.py \
		--question "What are the typical vibration characteristics of bearing outer race faults?" \
		--top_k 5

api:
	uvicorn app.api:app --host 0.0.0.0 --port 8000

gradio:
	python app/gradio_app.py

eval-retrieval:
	python scripts/05_evaluate_retrieval.py

eval-answer:
	python scripts/06_evaluate_answer.py

test:
	pytest -v
