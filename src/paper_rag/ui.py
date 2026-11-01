from __future__ import annotations

import os
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_rag.experiments.evaluate import evaluate, load_records

import requests
import streamlit as st


API_URL = os.getenv("PAPER_RAG_API_URL", "http://localhost:8000").rstrip("/")
EXAMPLES = [
    "比较 LightRAG 与 GraphRAG 的检索流程和适用场景。",
    "Self-RAG 如何判断是否需要检索，并如何评估生成结果？",
    "Adaptive-RAG 如何根据问题复杂度选择检索策略？",
]

st.set_page_config(page_title="Evidence Layout Lab", page_icon="📚", layout="wide")
st.markdown(
    """
    <style>
    :root {
      --ink: #17212b;
      --muted: #667482;
      --line: #dfe5e8;
      --surface: #f5f7f7;
      --accent: #087e75;
      --accent-soft: #e7f4f1;
    }
    .stApp { background: #fbfcfc; color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(251,252,252,.92); }
    [data-testid="stSidebar"] { background: #f3f6f5; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.25rem; }
    .brand-mark { color: var(--accent); font-size: .76rem; font-weight: 750; letter-spacing: .08em; }
    .brand-title { color: var(--ink); font-size: 1.28rem; font-weight: 720; margin: .15rem 0 .2rem; }
    .brand-copy { color: var(--muted); font-size: .86rem; line-height: 1.5; margin-bottom: 1.1rem; }
    .section-label { color: #56636e; font-size: .72rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; margin: 1.2rem 0 .45rem; }
    .status-panel { border: 1px solid var(--line); background: white; border-radius: 8px; padding: .75rem .85rem; margin: .5rem 0; }
    .status-line { display:flex; justify-content:space-between; gap:.5rem; padding:.27rem 0; font-size:.82rem; }
    .status-name { color: var(--muted); }
    .status-value { color: var(--ink); font-weight: 650; text-align:right; overflow-wrap:anywhere; }
    .status-ok { color: #087e75; }
    .status-bad { color: #b5473f; }
    .page-kicker { color: var(--accent); font-size:.78rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }
    .page-title { color:var(--ink); font-size:1.85rem; font-weight:740; line-height:1.18; margin:.15rem 0 .35rem; }
    .page-subtitle { color:var(--muted); font-size:.96rem; margin-bottom:1.1rem; }
    .answer-label { color:var(--accent); font-size:.74rem; font-weight:750; letter-spacing:.07em; text-transform:uppercase; margin-bottom:.3rem; }
    .st-key-answer_output { background:white; border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:7px; padding:1rem 1.15rem; line-height:1.75; }
    .source-index { color:var(--accent); font-weight:750; }
    div[data-testid="stMetric"] { border:1px solid var(--line); background:white; border-radius:7px; padding:.55rem .75rem; }
    div[data-testid="stMetricLabel"] { color:var(--muted); }
    div[data-testid="stMetricValue"] { font-size:1.28rem; line-height:1.35; }
    div[data-testid="stForm"] { border:1px solid var(--line); border-radius:8px; background:white; padding:1rem 1rem .75rem; }
    [data-testid="stFormSubmitButton"] button { background:var(--accent); border-color:var(--accent); color:white; }
    [data-testid="stFormSubmitButton"] button:hover { background:#06665f; border-color:#06665f; color:white; }
    button[kind="secondary"] { border-color:var(--line); }
    @media (max-width: 760px) {
      .page-title { font-size:1.5rem; }
      div[data-testid="stMetric"] { padding:.45rem .55rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_health() -> tuple[dict | None, str | None]:
    try:
        response = requests.get(f"{API_URL}/health", timeout=4)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)



page = st.sidebar.radio("页面", ["论文问答", "实验对比", "失败案例"])
if page != "论文问答":
    st.title(page)
    root = Path(__file__).resolve().parents[2]
    result_dir = st.sidebar.text_input("结果目录", str(root / "results" / "smoke"))
    try:
        directory = Path(result_dir).expanduser()
        manifest = json.loads((directory / "manifest.json").read_text())
        records = load_records(directory)
        st.caption("批量实验由命令行执行；此页面只读取已落盘结果。")
        expected = manifest["config"]["expected_records"]
        st.caption(f"已保存 {len(records)} / {expected} 次记录（错误记录也计入）。")
        if len(records) != expected:
            st.warning("实验尚未完整运行，下表是部分结果。")
        with st.expander("实验配置与环境"):
            st.json(manifest)
        if not records:
            st.info("尚无已保存的逐次记录。")
            st.stop()
        # Always recalculate from raw records; use saved report only for manual reviews.
        reviews = {}
        saved = directory / "evaluation.json"
        if saved.exists():
            for detail in json.loads(saved.read_text()).get("details", []):
                if detail.get("human_review"):
                    reviews[detail["run_id"]] = detail["human_review"]
        report = evaluate(records, reviews)
        if manifest["config"]["allow_unreviewed"]:
            st.warning("未人工核验的演示运行，不能作为正式研究结论。")
        if page == "实验对比":
            st.dataframe(report["summary"], use_container_width=True)
            st.caption(report["notes"])
            st.subheader("配对结果变化")
            st.dataframe(report["paired_changes"], use_container_width=True)
            st.download_button("下载完整评分", json.dumps(report, ensure_ascii=False, indent=2), "evaluation.json", "application/json")
        else:
            details = {s["run_id"]: s for s in report["details"]}
            only_failed = st.checkbox("只看有失败或条件间正确性不同的题", value=True)
            task_ids = sorted({r["task"]["id"] for r in records})
            if only_failed:
                task_ids = [tid for tid in task_ids if any(not details[r["run_id"]]["answer_correct"] for r in records if r["task"]["id"] == tid)]
            if not task_ids:
                st.info("没有符合筛选条件的案例。")
                st.stop()
            tid = st.selectbox("题目", task_ids)
            selected = [r for r in records if r["task"]["id"] == tid]
            options = sorted({(r["rotation"], r["reverse"]) for r in selected})
            rotation, reverse = st.selectbox("排列（轮转、反向）", options)
            st.write(selected[0]["task"]["question"])
            st.caption(f"标准答案：{selected[0]['task']['answer']}")
            cols = st.columns(3)
            for col, condition in zip(cols, ("necessary", "concentrated", "dispersed")):
                with col:
                    st.subheader(condition)
                    matches = [r for r in selected if r["condition"] == condition and r["rotation"] == rotation and r["reverse"] == reverse]
                    if not matches:
                        st.caption("此排列无记录（必要证据只运行轮转 0）。")
                        continue
                    r = matches[0]
                    ctx = r.get("context") or {}
                    st.json({k:ctx.get(k) for k in ("order", "required_positions", "input_tokens", "evidence_token_offsets")})
                    st.text(ctx.get("text", "上下文构造失败"))
                    st.code(r.get("generation", {}).get("answer", r.get("error", "")))
                    st.json(details[r["run_id"]])
                    with st.expander("完整原始记录"):
                        st.json(r)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"无法读取结果：{exc}")
    st.stop()

health, health_error = get_health()

with st.sidebar:
    st.markdown('<div class="brand-mark">RESEARCH WORKSPACE</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-title">Evidence Layout Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-copy">在论文证据中检索、比较与归纳研究结果。</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">运行状态</div>', unsafe_allow_html=True)
    if health:
        device = health.get("device", "unknown")
        gpu_ready = device == "cuda"
        state_label = "GPU 就绪" if gpu_ready else "CPU 模式"
        state_class = "status-ok" if gpu_ready else ""
        embedding_model = health.get("embedding_model") or "TF-IDF"
        llm_model = health.get("llm_model") or "未配置"
        reranker = health.get("reranker_model") or "未启用"
        st.markdown(
            f"""
            <div class="status-panel">
              <div class="status-line"><span class="status-name">API</span><span class="status-value status-ok">在线</span></div>
              <div class="status-line"><span class="status-name">设备</span><span class="status-value {state_class}">{device} · {state_label}</span></div>
              <div class="status-line"><span class="status-name">论文片段</span><span class="status-value">{health.get('chunks', 0):,}</span></div>
              <div class="status-line"><span class="status-name">Embedding</span><span class="status-value">{embedding_model}</span></div>
              <div class="status-line"><span class="status-name">Reranker</span><span class="status-value">{reranker}</span></div>
              <div class="status-line"><span class="status-name">生成模型</span><span class="status-value">{llm_model}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-panel"><span class="status-bad">API 离线</span><br><span class="status-name">请启动端口 8000 的 API 服务。</span></div>',
            unsafe_allow_html=True,
        )
        with st.expander("连接详情"):
            st.code(health_error or "未知错误")

    st.markdown('<div class="section-label">检索设置</div>', unsafe_allow_html=True)
    top_k = st.slider("返回证据数", min_value=1, max_value=10, value=5)

    st.markdown('<div class="section-label">试试这些问题</div>', unsafe_allow_html=True)
    selected_example = st.selectbox("示例问题", ["选择一个示例…", *EXAMPLES], label_visibility="collapsed")
    if selected_example != "选择一个示例…":
        st.session_state["paper_rag_question"] = selected_example

    st.divider()
    st.caption("来源范围：Zotero 文献库 · RAG / Agent")

st.markdown('<div class="page-kicker">PAPER KNOWLEDGE BASE</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">科研论文问答</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">输入研究问题，查看模型回答及其对应的论文证据。</div>', unsafe_allow_html=True)

with st.form("paper_query", border=False):
    question = st.text_area(
        "研究问题",
        key="paper_rag_question",
        placeholder="例如：比较 LightRAG 与 GraphRAG 的检索流程和适用场景。",
        height=112,
        label_visibility="collapsed",
    )
    left, right = st.columns([1, 5])
    with left:
        submitted = st.form_submit_button("检索并回答", icon=":material/search:", type="primary", use_container_width=True)
    with right:
        st.caption("支持论文问答、方法比较和研究总结；回答将附带来源页码。")

if submitted:
    if not question.strip():
        st.warning("先输入一个研究问题。")
    else:
        try:
            st.session_state.pop("paper_rag_result", None)
            with st.spinner("正在检索论文并组织证据…"):
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": question, "top_k": top_k},
                    timeout=600,
                )
                response.raise_for_status()
                result = response.json()
            st.session_state["paper_rag_result"] = result
            st.session_state["paper_rag_last_question"] = question
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"推理服务返回错误：{detail}")
        except requests.RequestException as exc:
            st.error(f"无法连接推理服务：{exc}")

result = st.session_state.get("paper_rag_result")
if result:
    st.divider()
    st.markdown('<div class="answer-label">ANSWER</div>', unsafe_allow_html=True)
    st.markdown(f"**{st.session_state.get('paper_rag_last_question', '')}**")

    gpu = result.get("gpu") or {}
    metric_cols = st.columns(4)
    metric_cols[0].metric("任务", {"compare": "论文比较", "summarize": "研究总结", "question_answering": "论文问答"}.get(result["agent"]["task"], result["agent"]["task"]))
    metric_cols[1].metric("推理设备", result.get("device", "未知"))
    metric_cols[2].metric("响应时间", f"{result.get('elapsed_seconds', 0):.2f} 秒")
    metric_cols[3].metric("GPU 已分配显存", f"{gpu['allocated_mb']:.0f} MB" if gpu else "未启用")

    with st.container(border=True, key="answer_output"):
        st.markdown(result.get("answer", "没有生成回答。"))

    if gpu:
        st.caption(
            f"{gpu.get('device_name', 'GPU')} · HIP {gpu.get('hip_version') or '可用'} · "
            f"显存可用 {gpu.get('free_mb', 0):.0f} / {gpu.get('total_mb', 0):.0f} MB"
        )

    with st.expander("Agent 执行路径"):
        st.write(" → ".join(result.get("agent", {}).get("steps", [])))

    sources = result.get("sources", [])
    st.subheader(f"引用证据 · {len(sources)} 条")
    if not sources:
        st.info("没有检索到匹配证据。可以换一种表述，或调整侧栏中的证据数量。")
    for number, source in enumerate(sources, start=1):
        title = source.get("paper_title", "未命名论文")
        page = source.get("page", "?")
        score = source.get("score", 0)
        with st.expander(f"[{number}]  {title}  ·  p.{page}  ·  {score:.3f}"):
            st.caption(f"{source.get('source_file', '')} · 片段 {source.get('chunk_id', '')}")
            st.write(source.get("text", ""))
            if source.get("retrieval_score") is not None:
                st.caption(f"初始检索分数 {source['retrieval_score']:.3f} · 重排分数 {score:.3f}")
else:
    st.divider()
    empty_left, empty_main, empty_right = st.columns([1, 6, 1])
    with empty_main:
        st.info("输入问题后开始检索。结果将显示在这里，并附上可核对的论文原文片段。")

