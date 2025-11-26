# DS-LoRA & SLSD: Continual Instruction Tuning Framework

本项目实现了硕士论文提出的 **Dual-Speed LoRA（DS-LoRA）** 与  
**Stability-aware Lightweight Self-Distillation（SLSD）**，用于解决持续指令微调
（Continual Instruction Tuning, CIT）中的：

- **虚假遗忘 (Illusory Forgetting)**
- **知识遗忘 (Forgetting)**

该仓库包含完整的工程实现、训练脚本、数据准备流程与可视化工具。

---

## 🚀 项目亮点（论文方法简介）

### 🔹 方法一：Dual-Speed LoRA（DS-LoRA）
用于在持续训练中同时保持 **稳定性 + 可塑性**。

核心思想：

1. **冻结底层 Transformer 层**
   - 保持基座模型对齐性、安全性、语言风格稳定
   - 避免假遗忘（风格漂移）

2. **在高层为每个线性层注入 Slow LoRA + Fast LoRA 双分支**
   - Slow LoRA：任务共享，小学习率 → 保持长期稳定记忆  
   - Fast LoRA：任务独立，大学习率 → 快速适应新任务  
   - 最终权重：  
     \[
     W = W_0 + \Delta W_{\text{slow}} + \Delta W_{\text{fast}}
     \]

3. **训练策略**
   - Slow 分支：小 lr（例如 `1e-5`）
   - Fast 分支：大 lr（例如 `5e-5`）
   - LoRA rank = 8，dropout = 0.05

DS-LoRA 解决：

- 🎯 不破坏旧任务性能  
- 🚀 快速适应新任务  
- 💡 避免虚假遗忘（style shift）

---

### 🔹 方法二：SLSD（轻量级稳定性自蒸馏）
进一步保证训练过程中 **输出风格保持一致**。

核心做法：

1. 使用当前任务的少量样本作为 probe buffer（100~500 条）
2. 用上一阶段模型生成 teacher logits（只前向一次）
3. 用输出熵筛选“代表旧风格的难忘样本”
4. **蒸馏仅作用于 Slow LoRA 分支**  
   - Slow LoRA：supervised loss + KD  
   - Fast LoRA：仅 supervised loss  

SLSD 在保证训练效率的同时显著减少风格漂移。

---

## 📂 项目结构
dslora_project/
│
├── configs/
│ └── base_config.py # 全局配置（路径、LoRA 参数、lr 等）
│
├── data/ # 五个 CIT 任务数据集（jsonl）
│ ├── T1_general.jsonl
│ ├── T2_math.jsonl
│ ├── T3_code.jsonl
│ ├── T4_tool.jsonl
│ └── T5_safety.jsonl
│
├── logs/ # 训练可视化 log（自动生成）
│
├── models/
│ └── ds_lora.py # DS-LoRA 核心实现
│
├── prepare_datasets.py # 下载并预处理五个任务数据集
├── utils_data.py # 数据处理、collate、tokenizer 等
│
├── train_single_task.py # 单任务训练（测试 DS-LoRA）
├── train_slsd_seq.py # 持续任务序列训练（DS-LoRA + SLSD）
│
├── plot_loss.py # loss 可视化脚本
│
└── checkpoints/ # 保存模型（自动生成）


## ⚙️ 环境安装

建议使用 conda：

```bash
conda create -n dslora python=3.10 -y
conda activate dslora
安装 PyTorch（CUDA 11）
bash
复制代码
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
安装其余依赖
bash
复制代码
pip install transformers datasets tqdm matplotlib accelerate
🧪 数据集准备
运行：

bash
复制代码
python prepare_datasets.py
脚本会自动生成 5 个 CIT 任务的数据集：

通用任务（T1）

数学（T2）

代码（T3）

工具调用（T4）

安全（T5）

🔥 训练流程
▶ 1. 单任务训练（验证 DS-LoRA）
示例：

bash
复制代码
python train_single_task.py --task T1_general
其他任务：

bash
复制代码
python train_single_task.py --task T2_math
python train_single_task.py --task T3_code
...
输出包括：

logs/single_*.jsonl

checkpoints/single_*

▶ 2. 持续训练（DS-LoRA + SLSD）
bash
复制代码
python train_slsd_seq.py
训练顺序：

nginx
复制代码
T1 → T2 → T3 → T4 → T5
每阶段自动评估已见任务，记录到：

bash
复制代码
logs/seq_eval_loss.jsonl
模型保存路径：

bash
复制代码
checkpoints/seq_T1_general/
checkpoints/seq_T2_math/
...
📈 可视化
单任务 loss 曲线：
bash
复制代码
python plot_loss.py single T1_general
输出文件：

bash
复制代码
logs/single_T1_general_train_loss.png
序列任务 loss 曲线：
bash
复制代码
python plot_loss.py seq
输出文件：

bash
复制代码
logs/seq_eval_loss.png
❗ 常见问题
Loss 出现 NaN？
使用 float32（已默认）

降低学习率：

python
复制代码
lr_slow = 1e-5
lr_fast = 5e-5
LoRA dropout 保持 0.05

增加 NaN 检查及时跳过坏样本

CUDA OOM？
降低 batch size

提升 gradient_accumulation_steps

缩短 max_seq_length

🙏 致谢
本项目基于：

HuggingFace Transformers

PyTorch

LoRA（Hu et al.）

你的硕士毕业论文所提出的方法

欢迎引用、扩展与改进。
