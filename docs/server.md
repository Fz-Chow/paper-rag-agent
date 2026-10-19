# AUP Learning Cloud 运行说明

## 目标环境

项目按 AUP Learning Cloud 的单节点 GPU 开发环境设计。服务器启动时填写 GitHub 仓库地址，进入工作空间后运行项目脚本。

## 首次检查

```bash
python scripts/check_runtime.py
```

重点观察：

- `CUDA available` 在 AMD PyTorch 环境中可能通过 HIP 后端工作；
- `HIP version` 是否有值；
- `rocminfo` 是否能列出 `gfx1151` 设备；
- Python 环境是否已经提供 PyTorch、ROCm 和模型运行时。

## 启动基线版本

```bash
pip install -r requirements.txt
python scripts/build_index.py
uvicorn paper_rag.api:app --app-dir src --host 0.0.0.0 --port 8000
```

## GPU 推理配置

先确认平台镜像中已有与 ROCm 匹配的 PyTorch/推理框架，再配置：

```bash
export PAPER_RAG_DEVICE=rocm
export PAPER_RAG_LLM_PROVIDER=openai_compatible
export PAPER_RAG_LLM_BASE_URL=http://127.0.0.1:8001/v1
export PAPER_RAG_LLM_MODEL=<服务器上的模型名>
```

不应在仓库中保存 API Key、模型权重或服务器访问凭据。若 GPU 后端不可用，保留 `PAPER_RAG_DEVICE=cpu` 并运行基线检索。

## 相关官方文档

- ROCm 文档：https://rocm.docs.amd.com/en/latest/
- AUP Learning Cloud：https://amdresearch.github.io/aup-learning-cloud/
- 单节点安装：https://amdresearch.github.io/aup-learning-cloud/installation/single-node.html

