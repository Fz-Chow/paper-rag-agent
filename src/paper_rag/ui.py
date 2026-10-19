from __future__ import annotations

import requests
import streamlit as st

st.set_page_config(page_title="PaperRAG-Agent", layout="wide")
st.title("PaperRAG-Agent")
st.caption("科研论文检索增强问答：回答仅应基于检索到的论文证据。")
question = st.text_area("问题", placeholder="例如：LightRAG 和 GraphRAG 的检索结构有什么区别？")
if st.button("检索并回答", type="primary") and question.strip():
    try:
        result = requests.post("http://localhost:8000/query", json={"question": question}, timeout=120).json()
        st.write(result["answer"])
        st.subheader("证据片段")
        for source in result["sources"]:
            with st.expander(f"{source['paper_title']} · 第 {source['page']} 页 · {source['score']}"):
                st.write(source["text"])
    except requests.RequestException as exc:
        st.error(f"API 未启动：{exc}")

