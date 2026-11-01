# AUP 受控实验操作指南

本指南面向已获得《AI技术科研实践应用》课程 AUP GPU 环境的用户，不需要自行部署 AUP 平台。先跑通单模型、双跳控制实验，再考虑真实检索聚合。本轮尚未连接用户的 AUP 实例执行以下命令。

## 1. 更新代码

在 AUP 的 GPU 环境中打开 Terminal。如果已有旧项目：

```bash
cd /home/jovyan/paper-rag-agent
git status --short
git pull --ff-only origin main
```

有本地改动时先保留并处理自己的改动，不要强制覆盖。首次下载则执行：

```bash
cd /home/jovyan
git clone https://github.com/Fz-Chow/paper-rag-agent.git evidence-layout-lab
cd evidence-layout-lab
```

后续命令都在项目根目录运行。上面的远端名称暂时保留，作者之后手动更名时再调整 clone 地址与 origin。AUP 官方建议把需保留的文件放在用户工作目录；实际持久化和配额以课程实例为准，重要结果另行下载备份。[平台与存储说明](https://amdresearch.github.io/aup-learning-cloud/user-guide/platform-basics.html)

## 2. 检查镜像并保留 ROCm PyTorch

```bash
python3 --version
python3 scripts/check_runtime.py --require-amd --output results/runtime-before-install.json
```

命令必须成功，且环境记录包含 HIP 版本、GPU 名称、GPU 可用和真实张量乘法检查。若失败，先确认启动的是课程 GPU 镜像和资源，不要直接 pip 重装 PyTorch。ROCm 沿用 `torch.cuda` 接口，因此日志显示 cuda 不等于 NVIDIA。[PyTorch 官方说明](https://docs.pytorch.org/docs/stable/notes/hip.html)

推荐隔离项目依赖，同时沿用镜像 PyTorch。项目不添加新依赖，以下安装的是已有依赖文件。用约束保留当前 torch 版本，避免间接依赖替换它：

```bash
python3 -m venv --system-site-packages .venv-aup
source .venv-aup/bin/activate
python3 -c 'import torch; from pathlib import Path; Path("results/torch-constraint.txt").write_text("torch==" + torch.__version__ + "\n")'
python3 -m pip install -r requirements.txt -c results/torch-constraint.txt
python3 scripts/check_runtime.py --require-amd --output results/runtime.json
```

若版本约束冲突，停止安装并保存错误，核对当前镜像与现有依赖，不通过升级 torch 绕过。不要把本地 macOS 的虚拟环境上传服务器。每个新终端需重新激活 AUP 虚拟环境；本指南环境名与旧本地 `.venv` 分开。

## 3. 配置一个生成模型

```bash
export PAPER_RAG_DEVICE=rocm
export PAPER_RAG_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct
export PAPER_RAG_MAX_NEW_TOKENS=128
export PAPER_RAG_MAX_INPUT_TOKENS=4096
unset PAPER_RAG_EMBEDDING_MODEL PAPER_RAG_RERANKER_MODEL
```

也可把模型变量设为服务器上已有、完整的本地模型目录。网络模型在首次运行时下载，具体内存、磁盘和许可见 [官方模型卡](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)。先确认配额，不同时启动 API、向量模型与批量实验以免互相占用资源。模型已缓存时不会重复下载全部权重。

本轮未核实用户 AUP 当前可用内存，不能保证给定模型在其上运行成功。OOM 时保留失败记录，检查课程资源；若改用更小模型或改变预算，应新建实验结果目录。

## 4. 运行最小真实实验并验收

```bash
python3 scripts/run_experiment.py --output results/smoke --limit 1 --allow-unreviewed --require-gpu
python3 scripts/verify_experiment.py results/smoke
```

一题共 14 次实际生成。第一条命令有运行错误时返回非零，但保留逐次记录。第二条检查所有预期调用已保存且成功、每次生成在 HIP GPU 上执行、峰值内存大于零、提示词与测量值一致，以及集中/分散使用同一材料和相同输入 token 数。验收通过只证明运行链路与控制条件，不要求模型回答全部正确，更不证明研究假设成立。

查看 `results/smoke/records/` 中实际回答与 `evaluation.json`。示例数据未人工核验，评分只能作流程演示。若集中/分散出现 token 数不一致，程序拒绝配对；核对实际 tokenizer 与材料边界，调整探索集设计后用新目录重新开始，不用截断或给一方补答案相关文本来“凑齐”。

可另开终端观察实际设备负载：

```bash
watch -n 2 rocm-smi
```

`rocm-smi` 缺失时使用课程提供的资源监控；不要仅凭 GPU 可见就认定 LLM 已参与推理。逐次生成记录中的设备、HIP 与峰值内存才是本项目验收依据。PyTorch 峰值不是整机内存或整卡占用。

## 5. 恢复与备份

中断后重新激活环境、重新 export 相同变量，再原样运行相同实验命令。已有成功调用会跳过，未落盘调用重新运行。可以在小规模演示时中断一次再恢复，确认最终恰好 14 条记录，并重新执行验收脚本。

错误记录默认保留；需要显式重试时：

```bash
python3 scripts/run_experiment.py --output results/smoke --limit 1 --allow-unreviewed --require-gpu --retry-errors
```

重试前的错误归档到 `attempts/`。修改数据、代码、模型、预算、划分、种子或环境后，不要沿用旧目录。环境未变时恢复有效；会话重建后系统版本变化可能触发拒绝恢复，需保存原目录、改用新目录运行并分开分析。

```bash
tar -czf evidence-layout-smoke.tar.gz results/smoke results/runtime.json
```

从 JupyterLab 文件浏览器下载压缩包。结果、模型缓存和虚拟环境都不随 GitHub 上传；会话结束后进程停止，用户卷是否持久化由课程设置决定。[JupyterLab 文件与终端指南](https://amdresearch.github.io/aup-learning-cloud/user-guide/jupyterlab-guide.html)

## 6. 查看实验页面

实验跑完释放模型进程后，在项目根目录启动：

```bash
source .venv-aup/bin/activate
python3 -m streamlit run src/paper_rag/ui.py --server.address 0.0.0.0 --server.port 8501
```

页面左侧选择“实验对比”或“失败案例”，结果目录填 `results/smoke` 的服务器绝对路径，也可使用项目根目录下的相对路径。不需要启动 API。页面展示当前已落盘记录，记录不足会提示结果不完整；人工引用核验后再重新评分，刷新页面查看。

Code Server 使用 `PORTS` 面板转发 8501，打开平台生成的访问地址；JupyterLab 使用课程实例实际提供的端口代理入口，不能假定所有环境都有相同 URL。仅监听 0.0.0.0 不会自动生成公网地址，不要把登录 token 放进仓库。[Code Server 官方端口说明](https://amdresearch.github.io/aup-learning-cloud/user-guide/code-server-guide.html)

## 7. 保留论文问答

实验与问答分时运行。要体验原有论文问答，在 API 所在终端设置模型变量并启动：

```bash
source .venv-aup/bin/activate
export PAPER_RAG_DEVICE=rocm
export PAPER_RAG_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct
python3 scripts/build_index.py
PYTHONPATH=src python3 -m uvicorn paper_rag.api:app --host 0.0.0.0 --port 8000
```

另开终端启动第 6 节界面，选择“论文问答”。默认 TF-IDF 检索在 CPU 上，生成在所选设备上；可选向量检索与重排变量保留，但不是第一版控制实验组成部分。问答首次请求才加载生成模型；健康检查只能说明配置与设备状态，不能替代真实请求验证。

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/query -H 'Content-Type: application/json' -d '{"question":"Self-RAG 如何使用检索证据？","top_k":3}'
```

检查回答、来源编号、页码、生成 token 和 GPU 指标。普通问答没有标准必要证据标注，不计算其必要证据覆盖率。

## 8. 从演示进入正式比较

1. 按任务集说明建立并人工核验独立双跳任务；同事实组合与改写放入同一划分。
2. 只用探索集调整材料、规则和提示；保留验证集，冻结后再执行。
3. 固定模型实际 revision、数据、预算与代码，另建正式输出目录，移除跳过核验选项。
4. 用引用核验示例对真实输出逐条标注支持性，记录错误原因并重新评分。
5. 报告逐题配对变化、运行失败和不支持假设的案例；不要将排列当独立样本，不用演示题量推出普遍结论。

完成这条链后再扩展真实检索和聚合，不能将集中条件的标准标签当作实际系统可获得的信息。
