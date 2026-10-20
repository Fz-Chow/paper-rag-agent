# PaperRAG-Agent

面向科研论文的可解释 RAG Agent。项目从 Zotero 选取 RAG/Agent 论文，支持 PDF 导入、GPU 向量检索、可选重排、本地 LLM 回答和来源引用。

## 当前版本

- PDF 文本解析与段落分块
- TF-IDF CPU 基线；配置 Embedding 后使用 ROCm GPU 向量检索
- 可选 CrossEncoder 重排和本地 Hugging Face LLM
- Agent 根据问题选择问答、总结或比较路径
- FastAPI API 与 Streamlit 界面
- LLM 后端通过环境变量配置，不把模型或密钥提交到仓库
- 设备配置支持 `auto`、`cpu`、`rocm`，ROCm PyTorch 使用 PyTorch 的 `cuda` 设备接口

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

在已有 AUP ROCm PyTorch 环境中启用 GPU 推理，不要重装 PyTorch。建议模型：`BAAI/bge-m3` 做多语言 Embedding、`BAAI/bge-reranker-v2-m3` 做重排、`Qwen/Qwen2.5-3B-Instruct` 做回答生成。模型首次下载需要网络和磁盘空间。项目依赖不声明 PyTorch，避免替换 AUP 镜像的 ROCm 版本。

```bash
python3 -m pip install -r requirements.txt
export PAPER_RAG_DEVICE=rocm
export PAPER_RAG_EMBEDDING_MODEL=BAAI/bge-m3
export PAPER_RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
export PAPER_RAG_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct
python3 scripts/build_index.py
PYTHONPATH=src python3 -m uvicorn paper_rag.api:app --host 0.0.0.0 --port 8000
```

API 启动时会将论文片段编码到 GPU；向量会缓存在 Git 忽略的 `data/index`，模型或论文变化后自动重建。LLM 在第一次提问时加载到 GPU。确认 `/health` 显示 `device: cuda`、`retrieval_backend: sentence-transformers`，并在页面查看 GPU 显存指标。若需 CPU 回退，设置 `PAPER_RAG_DEVICE=cpu` 并清空三个模型变量。

## Zotero 论文清单

首批论文见 `data/papers/manifest.json`。PDF 是从本机 Zotero 附件复制后的项目数据，未提交 Zotero 数据库、缓存或个人配置。

