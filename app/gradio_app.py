import requests
import gradio as gr


API_URL = "http://127.0.0.1:8000/query"


def format_sources(sources):
    if not sources:
        return "未返回引用来源。"

    lines = []
    for i, src in enumerate(sources, start=1):
        source_file = src.get("source_file", "")
        page = src.get("page_start", "")
        chunk_id = src.get("chunk_id", "")
        rerank_score = src.get("rerank_score", "")
        text = src.get("text", "").replace("\n", " ")[:500]

        lines.append(
            f"### [{i}] {source_file}, 第 {page} 页\n"
            f"- chunk_id: `{chunk_id}`\n"
            f"- rerank_score: `{rerank_score}`\n"
            f"- 片段预览：{text}\n"
        )

    return "\n".join(lines)


def query_rag(question, top_k):
    if not question.strip():
        return "请输入问题。", ""

    payload = {
        "question": question,
        "top_k": int(top_k)
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()

        answer = data.get("answer", "")
        sources = format_sources(data.get("sources", []))

        return answer, sources

    except Exception as e:
        return f"请求失败：{e}", ""


demo = gr.Blocks(title="Industrial Maintenance RAG QA")

with demo:
    gr.Markdown("# Industrial Maintenance RAG QA")
    gr.Markdown("面向轴承/工业设备运维知识库的 RAG 问答 Demo。")

    with gr.Row():
        with gr.Column(scale=2):
            question = gr.Textbox(
                label="用户问题",
                lines=4,
                placeholder="例如：What are the typical vibration characteristics of bearing outer race faults?"
            )
            top_k = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Top-K 引用片段数量"
            )
            submit = gr.Button("检索并回答", variant="primary")

        with gr.Column(scale=3):
            answer = gr.Textbox(
                label="模型回答",
                lines=12
            )
            sources = gr.Markdown(
                label="引用来源"
            )

    submit.click(
        fn=query_rag,
        inputs=[question, top_k],
        outputs=[answer, sources]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
