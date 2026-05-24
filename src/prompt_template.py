from typing import List, Dict


RAG_PROMPT_TEMPLATE = """你是工业设备运维知识库问答助手，面向维修工程师提供故障诊断和处理建议。

请严格根据【参考资料】回答用户问题。
如果参考资料中没有明确依据，请回答“知识库中未找到明确依据”，不要编造。

【用户问题】
{question}

【参考资料】
{context}

【回答要求】
1. 先给出直接结论；
2. 再列出诊断依据；
3. 给出维修或排查建议；
4. 最后列出引用来源，格式为：[文件名，第 x 页]；
5. 不要使用参考资料之外的事实。
"""


def format_context(chunks: List[Dict]) -> str:
    context_parts = []

    for idx, item in enumerate(chunks, start=1):
        source_file = item.get("source_file", "")
        page = item.get("page_start", "")
        chunk_id = item.get("chunk_id", "")
        text = item.get("text", "").strip()

        context_parts.append(
            f"【资料 {idx}】\n"
            f"来源文件：{source_file}\n"
            f"页码：第 {page} 页\n"
            f"chunk_id：{chunk_id}\n"
            f"正文：{text}\n"
        )

    return "\n".join(context_parts)


def build_rag_prompt(question: str, chunks: List[Dict]) -> str:
    context = format_context(chunks)
    return RAG_PROMPT_TEMPLATE.format(
        question=question,
        context=context
    )
