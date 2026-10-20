# AUP Learning Cloud 使用指南

本文面向已经可以登录 AUP Learning Cloud/JupyterHub 的项目使用者，说明如何在现有 GPU 开发环境中拉取、启动和观察 PaperRAG-Agent。AUP 单节点安装文档主要面向平台/节点部署；课程用户通常使用已创建的 Jupyter 环境，不需要自行安装整套 AUP 平台。

## 1. 服务器与工作目录

在课程指定的 GPU 开发环境中打开 Terminal。建议把代码、模型缓存和索引放在用户工作目录（当前环境示例为 `/home/jovyan`）下，并查看课程提供的磁盘配额和持久化说明。服务器到期会停止计算进程；请不要把唯一副本放在临时目录，也不要假设环境重置后未持久化的内容仍然存在。

```bash
cd /home/jovyan
git clone https://github.com/Fz-Chow/paper-rag-agent.git
cd paper-rag-agent
```

已经克隆过时直接进入目录并更新：

```bash
cd /home/jovyan/paper-rag-agent
git pull origin main
```

## 2. 检查 AUP GPU 与 Python 环境

```bash
python3 --version
python3 scripts/check_runtime.py
rocm-smi
rocminfo | head -80
```

本项目在这台课程环境的一次运行中观察到 AMD Ryzen AI Max+ PRO 395、Radeon 8060S Graphics、PyTorch `2.9.1+rocm7.13`、HIP `7.13.99004`，且 `torch.cuda.is_available()` 为 `True`。这是当时环境的实测记录，不代表每次分配到的实例或课程配额都相同。ROCm 版 PyTorch 使用 `torch.cuda` 兼容接口报告设备，因此这里看到 `CUDA available: True` 不表示服务器装有 NVIDIA GPU。

这类 AMD APU 的 GPU 与处理器属于同一 SoC；本项目没有检测或承诺固定的专用显存配额。一次 `rocm-smi` 空闲采样显示 GPU 利用率和 VRAM 使用率为 `0%`，只能说明采样时未见 GPU 工作，不能据此判断模型运行时可用内存上限。实际分配以 AUP 当前实例状态和课程配额为准。

如果 `python` 命令不存在，使用 `python3`。如果 `rocminfo`、PyTorch 或 GPU 不可用，先确认当前启动的是课程分配的 GPU 镜像/实例，再检查管理员提供的环境说明；不要用 pip 重装 PyTorch 来尝试修复 ROCm。

## 3. 安装项目依赖

本仓库的依赖文件刻意不声明 PyTorch，以保留 AUP 镜像自带的 ROCm 版本。先核对 PyTorch/HIP，再安装其余依赖：

```bash
python3 -m pip install -r requirements.txt
python3 -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())'
```

首次安装会下载 Python 包；配置模型后还会下载模型权重。模型缓存通常位于 `~/.cache/huggingface`，请结合用户磁盘配额检查空间。安装或下载失败时先保留完整错误信息，并确认网络、磁盘余量和课程镜像限制。

## 4. 选择推理配置

CPU 基线不下载深度学习模型，使用 TF-IDF 搜索论文证据。没有生成模型时，API 会返回证据并提示未配置 LLM。GPU 模式可分阶段启用：

```bash
export PAPER_RAG_DEVICE=rocm
export PAPER_RAG_EMBEDDING_MODEL=BAAI/bge-m3
```

需要重排和本地回答时，再配置：

```bash
export PAPER_RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
export PAPER_RAG_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct
export PAPER_RAG_MAX_NEW_TOKENS=512
```

这些是可选模型示例，不是 AUP 镜像预装组件。首次运行需要下载权重并占用磁盘、内存和显存；建议先验证单个模型。若 OOM，可先移除 reranker、使用较小生成模型或调低 `PAPER_RAG_MAX_NEW_TOKENS`，并在每次调整后重启 API 进程。若使用 `PAPER_RAG_DEVICE=cpu`，清空模型变量才能回到不加载模型的纯 TF-IDF 模式。

## 5. 构建索引并启动 API

切换到项目目录后构建/验证论文索引：

```bash
cd /home/jovyan/paper-rag-agent
python3 scripts/build_index.py
```

启动 API。保持这个 Terminal 运行：

```bash
PYTHONPATH=src python3 -m uvicorn paper_rag.api:app --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

GPU Embedding 在 API 初始化索引时加载；LLM 在首次提问时加载。`data/index/` 中的向量缓存会被 Git 忽略，并按论文和模型内容生成缓存键。更新论文或模型后，程序会生成新缓存；旧缓存可在确认不再需要后自行清理。

## 6. 启动界面与访问端口

另开一个 Terminal 启动 Streamlit：

```bash
cd /home/jovyan/paper-rag-agent
streamlit run src/paper_rag/ui.py --server.address 0.0.0.0 --server.port 8501
```

端口访问方式按所分配环境操作。AUP 官方 JupyterLab 指南说明可用 Terminal 并把重要文件保存到 `/home/jovyan`；Code Server 指南才明确描述 `PORTS` 面板和 forwarded address。若课程提供的是 JupyterLab，请使用该实例实际提供的服务代理/端口访问能力；若使用 Code Server，则按官方 Code Server 指南在 `PORTS` 面板打开 8501。环境未提供 Web 端口代理时，单纯监听 `0.0.0.0:8501` 不会自动生成浏览器可访问的 URL。不要把 Jupyter token、Hub URL 中的敏感参数或服务器凭据贴进仓库、截图或公开日志。Streamlit 默认调用同一节点上的 `http://localhost:8000`，因此 UI 和 API 应运行在同一个服务器环境；如需改 API 地址，设置 `PAPER_RAG_API_URL` 后重新启动 Streamlit。

页面左侧展示 API 状态、设备、论文片段数、Embedding、Reranker 和生成模型配置。回答区展示任务类型、推理设备、响应时间、GPU 已分配显存及引用证据。`/health` 的 `device` 可能显示 `cuda`，这是 ROCm PyTorch 采用的设备 API 名称；再结合 HIP 版本和 `rocm-smi` 判定 AMD GPU 是否可见。

## 7. 查看资源占用

可在第三个 Terminal 中检查 GPU、CPU、内存和磁盘：

```bash
watch -n 2 rocm-smi
```

```bash
top
free -h
df -h /home/jovyan
du -sh ~/.cache/huggingface data/index
```

推理页面中的“GPU 已分配显存”是 PyTorch 进程指标，不等同于整张 GPU 的总使用率。`rocm-smi` 用于观察设备级利用率和显存；`top`/`free` 用于观察主机 CPU 与内存；`df`/`du` 用于检查用户卷和模型/索引缓存。模型加载时资源上升是预期现象；没有查询时 GPU 利用率可能接近零。

## 8. 会话到期、停止和恢复

运行时限到达后，当前 API/UI 进程会停止，GPU 计算资源也会释放。本项目当前借助 GPU 执行 Embedding、可选重排和 LLM 推理，尚未实现模型微调/训练作业。重新登录后：

1. 确认项目目录与论文 PDF 仍在 `/home/jovyan/paper-rag-agent`。
2. 在 AUP 控制页按课程要求重新启动 GPU 开发环境。
3. 重新打开两个 Terminal，按第 5、6 节分别启动 API 和 Streamlit。
4. 打开 `/health` 或页面确认 API 在线、模型配置和 GPU 状态。

若项目目录不存在，说明当前工作目录未被保留或环境被重建，需要重新 clone 仓库。`data/index/` 可从论文和模型配置重新构建。大型模型缓存是否保留取决于课程用户卷与配额设置；若缓存不在，模型需要重新下载。GitHub 只保存已提交到仓库的代码与论文文件，不保存运行进程、GPU 状态、环境变量或忽略的缓存。

停止服务时，在各自 Terminal 使用 `Ctrl+C`。如运行时已断开，重新连接后可检查进程：

```bash
ps -u "$(whoami)" -o pid,cmd | grep -E 'uvicorn|streamlit' | grep -v grep
```

## 9. 常见问题

- `python: command not found`：使用 `python3`。
- 页面显示 API 离线：确认 API 在 8000 端口运行，UI 与 API 位于同一节点。
- 页面可开但回答提示未配置 LLM：确认 `PAPER_RAG_LLM_MODEL` 在启动 API 的同一 Terminal 中已设置，然后重启 API。
- GPU 显存显示未启用：先看 `/health`、`scripts/check_runtime.py`、`rocminfo` 和 `rocm-smi`；纯 TF-IDF 基线不会使用 GPU。
- 缺少 `transformers` 或 `sentence_transformers`：在正确环境中运行 `python3 -m pip install -r requirements.txt`，不要因此重装 PyTorch。
- 模型下载失败或磁盘不足：检查网络、用户卷配额和 `df -h /home/jovyan`；模型首次下载需联网。
- 进程启动时显示端口占用：用 `ps` 检查本用户进程，停止自己的旧服务或选择新端口，并同步调整 AUP 端口入口。
- 推理触发 OOM：先禁用重排，使用更小 LLM 或减少生成 token，再重新启动服务。

## 官方文档

- [AUP Learning Cloud 文档](https://amdresearch.github.io/aup-learning-cloud/)
- [AUP Platform Basics：环境时限与持久目录](https://amdresearch.github.io/aup-learning-cloud/user-guide/platform-basics.html)
- [AUP JupyterLab 用户指南](https://amdresearch.github.io/aup-learning-cloud/user-guide/jupyterlab-guide.html)
- [AUP Code Server 用户指南与端口转发](https://amdresearch.github.io/aup-learning-cloud/user-guide/code-server-guide.html)
- [AUP 单节点安装文档](https://amdresearch.github.io/aup-learning-cloud/installation/single-node.html)
- [ROCm 文档](https://rocm.docs.amd.com/en/latest/)
