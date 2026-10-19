# PaperRAG-Agent

面向科研论文的可解释 RAG Agent。项目从 Zotero 选取 RAG/Agent 论文，支持 PDF 导入、混合检索、来源引用和可选的 OpenAI-compatible LLM 后端。

## 当前版本

- PDF 文本解析与段落分块
- TF-IDF 基线检索，默认 CPU 可运行
- 可选关键词过滤和来源引用
- FastAPI API 与 Streamlit 界面
- LLM 后端通过环境变量配置，不把模型或密钥提交到仓库
- 设备配置支持 `auto`、`cpu`、`rocm`，便于 AUP Learning Cloud 的 AMD 环境适配

## 本地运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_index.py
uvicorn paper_rag.api:app --app-dir src --reload
```

另开终端运行界面：

```bash
streamlit run src/paper_rag/ui.py
```

## 云端运行

在 AUP Learning Cloud 选择 GPU 开发环境并填写 GitHub 仓库地址后，进入项目目录：

```bash
pip install -r requirements.txt
python scripts/check_runtime.py
python scripts/build_index.py
uvicorn paper_rag.api:app --app-dir src --host 0.0.0.0 --port 8000
```

当前基线检索不依赖 GPU。GPU 主要用于后续的 Embedding、Reranker 和本地 LLM 推理；设备检测脚本会记录 PyTorch/ROCm 状态，但不会强制安装特定 ROCm 版本。

## Zotero 论文清单

首批论文见 `data/papers/manifest.json`。PDF 是从本机 Zotero 附件复制后的项目数据，未提交 Zotero 数据库、缓存或个人配置。

