# Project Design

## 1. Project Overview

Industrial Maintenance RAG QA is a retrieval-augmented generation system for industrial equipment maintenance and bearing fault diagnosis.

The system parses industrial PDF documents, builds a searchable knowledge base, retrieves relevant evidence, reranks candidate chunks, and generates evidence-grounded answers with source citations.

## 2. System Architecture

~~~text
PDF Documents
    |
    v
PDF Parser
    |
    v
Page-level Documents JSONL
    |
    v
Sliding-window Chunker
    |
    v
Chunks JSONL
    |
    +-------------------------+
    |                         |
    v                         v
BGE-M3 Dense Embedding     BM25 Sparse Retrieval
    |                         |
    v                         v
Milvus Vector Search       Keyword Search
    |                         |
    +-----------+-------------+
                |
                v
        RRF Hybrid Fusion
                |
                v
        BGE Reranker
                |
                v
        Top Evidence Chunks
                |
                v
        Local Qwen Generation
                |
                v
Answer + Sources + Page + Chunk ID
~~~

## 3. Core Modules

### 3.1 Document Parsing

The parser reads PDF files from `data/raw_docs/` and converts them into page-level JSONL records. Each record keeps metadata such as file name, page number, language, text length, and scanned-page flag.

Output:

~~~text
data/parsed/documents.jsonl
~~~

### 3.2 Text Chunking

The chunker applies sliding-window segmentation to parsed page text. Each chunk preserves citation metadata including `source_file`, `page_start`, `page_end`, and `chunk_id`.

Output:

~~~text
data/chunks/chunks.jsonl
~~~

### 3.3 Vector Indexing

The system uses `BAAI/bge-m3` to encode chunks into dense vectors and stores them in Milvus. The Milvus collection stores embeddings and metadata fields for citation tracking.

Collection:

~~~text
industrial_maintenance_chunks
~~~

### 3.4 Hybrid Retrieval

The retrieval pipeline combines dense retrieval from Milvus, BM25 sparse retrieval from local chunks, RRF fusion, and BGE reranker.

### 3.5 Answer Generation

The answer generation module uses a local Qwen model. The prompt constrains the model to answer only based on retrieved evidence and cite source file, page number, and chunk ID.

## 4. Evaluation

The project includes retrieval and answer generation evaluation.

Retrieval metrics:

- Recall@5
- Recall@10
- MRR
- nDCG@5

Answer generation metrics:

- Answer Coverage
- Citation Presence Rate
- Hallucination Flag Rate

## 5. Deployment

The project runs on Windows + WSL2 + Docker. Milvus is deployed through Docker Compose, while FastAPI and Gradio run in the local Conda environment.

Main commands are wrapped in `Makefile`.

## 6. Current Status

The current MVP has completed PDF parsing, chunking, Milvus indexing, hybrid retrieval, reranking, local Qwen generation, FastAPI service, Gradio demo, retrieval evaluation, answer generation evaluation, and unit tests.
