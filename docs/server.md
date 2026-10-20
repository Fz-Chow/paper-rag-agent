# AUP Learning Cloud 运行说明

## 目标环境

项目按 AUP Learning Cloud 的单节点 GPU 开发环境设计。服务器启动时填写 GitHub 仓库地址，进入工作空间后运行项目脚本。

## 首次检查

```bash
python3 scripts/check_runtime.py
```

重点观察：

- `CUDA available` 在 AMD PyTorch 环境中可能通过 HIP 后端工作；
- `HIP version` 是否有值；
- `rocminfo` 是否能列出 `gfx1151` 设备；
- Python 环境是否已经提供 PyTorch、ROCm 和模型运行时。

## 安装与模型设置

```bash
python3 -m pip install -r requirements.txt
export PAPER_RAG_DEVICE=rocm
export PAPER_RAG_EMBEDDING_MODEL=BAAI/bge-m3
export PAPER_RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
export PAPER_RAG_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct
python3 scripts/build_index.py
```

`requirements.txt` 不安装或覆盖 PyTorch。AUP 镜像已提供 ROCm PyTorch 时，保留镜像版本。模型文件会缓存在当前用户的 Hugging Face 缓存目录，首次下载需要较多时间和网络流量。论文向量缓存在 Git 忽略的 `data/index`，重启服务后可复用。

## 启动服务

```bash
PYTHONPATH=src python3 -m uvicorn paper_rag.api:app --host 0.0.0.0 --port 8000
```

在 AUP `PORTS` 面板公开 8000。另开终端启动 Streamlit 并公开 8501：

```bash
streamlit run src/paper_rag/ui.py --server.address 0.0.0.0 --server.port 8501
```

`/health` 检查设备与模型配置；页面展示检索设备、请求耗时及 GPU 显存。`PAPER_RAG_DEVICE=cpu` 可切回 CPU 基线。不要在仓库提交模型权重、令牌或服务器凭据。

## 相关官方文档

- ROCm 文档：https://rocm.docs.amd.com/en/latest/
- AUP Learning Cloud：https://amdresearch.github.io/aup-learning-cloud/
- 单节点安装：https://amdresearch.github.io/aup-learning-cloud/installation/single-node.html

