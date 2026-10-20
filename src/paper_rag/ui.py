from __future__ import annotations

import os

import requests
import streamlit as st


API_URL = os.getenv("PAPER_RAG_API_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="PaperRAG-Agent", page_icon="📚", layout="wide")
st.title("PaperRAG-Agent")
st.caption("科研论文检索与证据问答")

try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    st.caption(
        f"{health['chunks']:,} 个论文片段 · {health['retrieval_backend']} · "
        f"设备：{health['device']} · LLM：{health['llm_model'] or '未配置'}"
    )
except requests.RequestException:
    st.warning("API 尚未连接。请先启动端口 8000 的 API 服务。")

with st.form("paper_query"):
    question = st.text_area(
        "研究问题",
        placeholder="例如：比较 LightRAG 与 GraphRAG 的检索流程和适用场景。",
        height=100,
    )
    top_k = st.slider("证据数量", min_value=1, max_value=10, value=5)
    submitted = st.form_submit_button("检索并生成回答", type="primary")

if submitted and question.strip():
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question, "top_k": top_k},
            timeout=600,
        )
        response.raise_for_status()
        result = response.json()
        metrics = st.columns(4)
        metrics[0].metric("任务类型", result["agent"]["task"])
        metrics[1].metric("设备", result["device"])
        metrics[2].metric("耗时", f"{result['elapsed_seconds']} s")
        gpu = result.get("gpu")
        metrics[3].metric("GPU 已分配显存", f"{gpu['allocated_mb']:.0f} MB" if gpu else "未启用")

        st.subheader("回答")
        st.markdown(result["answer"])
        with st.expander("Agent 执行步骤"):
            st.write(" → ".join(result["agent"]["steps"]))

        st.subheader("引用证据")
        for number, source in enumerate(result["sources"], start=1):
            with st.expander(
                f"[{number}] {source['paper_title']} · 第 {source['page']} 页 · 分数 {source['score']}"
            ):
                st.caption(f"文件：{source['source_file']} · 片段：{source['chunk_id']}")
                st.write(source["text"])
    except requests.RequestException as exc:
        st.error(f"请求失败：{exc}")

