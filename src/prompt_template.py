from typing import List, Dict


RAG_PROMPT_TEMPLATE = """You are an industrial maintenance knowledge-base QA assistant.

You must follow these rules:

1. Answer strictly based on the provided reference materials.
2. Use the same language as the user question.
3. Do not fabricate URLs, DOIs, file names, page numbers, or external sources.
4. Only cite source_file, page, and chunk_id from the provided reference materials.
5. If the references do not provide clear evidence, say: "No clear evidence was found in the knowledge base."
6. Cover the key diagnostic concepts, evidence, and maintenance suggestions when they are supported by the references.

User question:
{question}

Reference materials:
{context}

Answer format:

## Direct conclusion
Give a concise answer to the question.

## Diagnostic evidence
List evidence from the references. Each item must include citation in this format:
[source_file, page x, chunk_id=xxx]

## Maintenance or inspection suggestions
List maintenance, inspection, or troubleshooting suggestions if supported by the references.
If no clear maintenance suggestion is provided, say:
"No explicit maintenance suggestion was found in the references."

## Sources
Only list sources actually used in the answer:
- [source_file, page x, chunk_id=xxx]

Do not output any URL.
Do not cite sources outside the provided reference materials.
"""


def format_context(chunks: List[Dict]) -> str:
    context_parts = []

    for idx, item in enumerate(chunks, start=1):
        source_file = item.get("source_file", "")
        page = item.get("page_start", "")
        chunk_id = item.get("chunk_id", "")
        text = item.get("text", "").strip()

        context_parts.append(
            f"[Reference {idx}]\n"
            f"source_file: {source_file}\n"
            f"page: {page}\n"
            f"chunk_id: {chunk_id}\n"
            f"text: {text}\n"
        )

    return "\n".join(context_parts)


def build_rag_prompt(question: str, chunks: List[Dict]) -> str:
    context = format_context(chunks)
    return RAG_PROMPT_TEMPLATE.format(
        question=question,
        context=context
    )
