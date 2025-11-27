# DS-LoRA & SLSD: Continual Instruction Tuning Framework

本项目实现了硕士论文提出的：

- **Dual-Speed LoRA（DS-LoRA）**
- **Stability-aware Lightweight Self-Distillation（SLSD）**

用于解决持续指令微调（Continual Instruction Tuning, CIT）场景中的：
本项目实现了硕士论文提出的：

- **Dual-Speed LoRA（DS-LoRA）**
- **Stability-aware Lightweight Self-Distillation（SLSD）**

用于解决持续指令微调（Continual Instruction Tuning, CIT）场景中的：

- **虚假遗忘 (Illusory Forgetting)** —— 输出风格、对齐、安全性漂移  
- **知识遗忘 (Forgetting)** —— 旧任务性能下降
- **虚假遗忘 (Illusory Forgetting)** —— 输出风格、对齐、安全性漂移  
- **知识遗忘 (Forgetting)** —— 旧任务性能下降

该仓库包含完整的工程实现、训练脚本、数据准备流程与可视化工具。

---

## 🧠 0. 给未来 ChatGPT / 助手的“快速恢复摘要”

> 如果你是一个 AI 助手，用户说“请基于 README 帮我继续改代码”，  
> 请优先记住下面几点来恢复上下文：

- 项目核心是 **持续指令微调 (CIT)**，任务序列为：  
  **T1（General） → T2（Math） → T3（Code） → T4（Tool-Calling） → T5（Safety）**
- 方法一：**DS-LoRA**  
  - 冻结底层 Transformer 层  
  - 高层 Linear → `DSLoRALinear` (slow + fast 两个 LoRA 分支)  
  - Slow：任务共享，小 lr（长期记忆）  
  - Fast：任务特定，大 lr（快速适应新任务）  
- 方法二：**SLSD**  
  - 用上一阶段模型 + 当前任务少量样本构建 probe buffer  
  - 按熵筛选“风格代表样本”，只对 slow 分支做 KD  
  - fast 分支只走 supervised loss，保证可塑性  
- 模型：当前默认 **LLaMA 2 7B**（`meta-llama/Llama-2-7b-hf`），用 LoRA 做持续训练  
- 数据：  
  - `data/*.jsonl`：toy / 小规模调试数据  
  - `data/full/*_full.jsonl`：完整训练数据（正式实验使用）  
- 训练脚本：  
  - `train_single_task.py`：单任务 DS-LoRA  
  - `train_slsd_seq.py`：五任务序列，DS-LoRA + SLSD  
- 所有 LoRA / 学习率 / 数据路径等全局设置在：`configs/base_config.py`  
- `models/ds_lora.py`：包含 **DSLoRALinear + 注入 + 参数分组** 的所有细节  
- `utils_data.py`：使用 LLaMA-style `[INST] ... [/INST]` 模板，只监督输出部分的 token

---

## 🚀 1. 论文方法概述

### 🔹 1.1 Dual-Speed LoRA（DS-LoRA）

目标：**同时保持稳定性（不忘旧任务）和可塑性（快速学新任务），避免虚假遗忘**。

1. **冻结底层 Transformer 层**
   - 对 LLaMA 2 7B（32 层）通常冻结前 ~16 层
   - 保留基座模型的对齐性、安全性、语言风格

2. **高层 Linear: Slow + Fast 双 LoRA 分支**

对于每个被替换的 Linear 权重 \(W\)：

\[
W = W_0 + \Delta W_{\text{slow}} + \Delta W_{\text{fast}}
\]

- **Slow LoRA**
  - 所有任务共享
  - 学习率小（如 `1e-5`）
  - 负责长期稳定记忆

- **Fast LoRA**
  - 每个任务各自独立
  - 学习率大（如 `5e-5`）
  - 负责快速适应新任务

3. **训练中的角色分工**
   - 底层层（冻结）：保证风格、安全性不乱改
   - slow 分支：渐进累计多任务知识
   - fast 分支：专注当前任务的适配

---

### 🔹 1.2 SLSD（Stability-aware Lightweight Self-Distillation）

目标：**在任务序列训练中保持输出风格 & 对齐稳定**，防止“虚假遗忘”。

核心流程：

1. **Probe Buffer（每个任务约 100~500 条样本）**
   - 来自当前任务数据
   - 用上一阶段模型 \(\theta^{(t-1)}\) 前向一次，记录 logits
   - 之后不再调用 teacher（极轻量）

2. **按熵筛选“代表风格”样本**
   \[
   s(x) = H(p_{\theta^{(t-1)}}(\cdot|x))
   \]
   - 熵低 → 模型对该样本的输出非常自信 → 风格稳定 → 适合蒸馏

3. **只蒸馏 Slow 分支**
   - Slow：`L_slow = L_supervised + λ_KD * L_KD`
   - Fast：**只**用 supervised loss（不做 KD）  
   - 避免 KD 把 fast 分支也拉回旧任务，保留对新任务的可塑性

---

## 📂 2. 代码结构与文件详细说明

项目目录（逻辑结构）：

```text
dslora_project/
│
├── configs/
│   └── base_config.py
│
├── data/
│   ├── T1_general.jsonl       # toy / 小规模调试数据（可选）
│   ├── T2_math.jsonl
│   ├── T3_code.jsonl
│   ├── T4_tool.jsonl
│   ├── T5_safety.jsonl
│   └── full/
│       ├── T1_general_full.jsonl
│       ├── T2_math_full.jsonl
│       ├── T3_code_full.jsonl
│       ├── T4_tool_full.jsonl
│       └── T5_safety_full.jsonl
│
├── logs/
├── models/
│   └── ds_lora.py
│
├── prepare_datasets.py
├── utils_data.py
│
├── train_single_task.py
├── train_slsd_seq.py
│
├── plot_loss.py
└── checkpoints/
下面逐个文件解释。

📁 configs/base_config.py
核心配置类 BaseConfig，集中所有重要超参数与路径：

模型与 LoRA

model_name: 默认 "meta-llama/Llama-2-7b-hf"

lora_r, lora_alpha, lora_dropout

lora_target_modules: 默认 ("q_proj", "v_proj")

num_frozen_layers: 冻结的底层层数（如 16）

DS-LoRA 学习率

lr_slow: slow 分支学习率，如 1e-5

lr_fast: fast 分支学习率，如 5e-5

weight_decay

SLSD 超参

use_slsd: 是否启用 SLSD（序列训练时设为 True）

kd_lambda: KD loss 系数

probe_size_per_task: 每个任务 probe buffer 大小，如 500

entropy_threshold: 选入 buffer 的熵阈值

数据路径

use_toy_data: bool

True → 使用 data/T*_xxx.jsonl（小数据调试）

False → 使用 data/full/T*_full.jsonl（完整实验）

data_paths: 一个 dict，形如：

python
复制代码
data_paths = {
    "T1_general": {"toy": "data/T1_general.jsonl",
                   "full": "data/full/T1_general_full.jsonl"},
    ...
}
训练参数

max_seq_len: 例如 2048

per_device_batch_size: 通常为 1（7B 模型显存限制）

gradient_accumulation_steps: 用于模拟大的 batch

num_epochs

save_dir: checkpoint 保存目录

想调实验：优先改这个文件。

📁 prepare_datasets.py
负责 准备五个任务的数据集，生成统一格式的 jsonl：

json
复制代码
{"instruction": "...", "input": "...", "output": "..."}
当前版本主要生成 full 数据集：

data/full/T1_general_full.jsonl

源自：

tatsu-lab/alpaca

databricks/databricks-dolly-15k

data/full/T2_math_full.jsonl

源自 openai/gsm8k 的 train 集（数学推理）

data/full/T3_code_full.jsonl

源自 sahil2801/CodeAlpaca-20k（代码生成）

data/full/T4_tool_full.jsonl

Mini-ToolBench 风格工具调用数据

从类似 openai-function-calling 风格数据集中采样

每条 instruction 是用户自然语言请求
output 是工具调用（tool_calls / function_call）的 JSON 字符串

data/full/T5_safety_full.jsonl

源自 Anthropic/hh-rlhf 的安全对齐数据

使用 prompt + chosen 构造安全回答样本

注意：

full 数据全部写入 data/full/，已经通过 .gitignore 忽略，不会被 git 跟踪。

toy 版 data/T*_xxx.jsonl 用于快速本地测试，可自行从 full 采样得到。

📁 utils_data.py
数据加载与 collate 逻辑。

InstructionDataset
读取给定的 .jsonl 文件（full 或 toy）

对每条样本构造 LLaMA 风格指令模板：

text
复制代码
<s>[INST] {instruction}
{input} [/INST] {output}</s>
只监督 output 部分的 token：

在 prompt 部分（<s>[INST] ... [/INST]）的 labels 设置为 -100

使模型专注学习“如何回答”，而不是记 prompt 文本

返回字段：

input_ids

attention_mask

labels（已 mask 好）

collate_fn
对 batch 中序列做 pad：

input_ids 用 pad_token_id pad

attention_mask pad 为 0

labels pad 为 -100

返回一个适配 AutoModelForCausalLM 的字典

对 LLaMA / Mistral / Qwen 等所有 CausalLM 都适用。

📁 models/ds_lora.py
核心方法文件：DS-LoRA 的全部实现。

class DSLoRALinear(nn.Module)
替换原始 nn.Linear：

𝑦
=
𝑥
𝑊
⊤
+
Δ
𝑊
slow
(
𝑥
)
+
Δ
𝑊
fast
(
𝑥
)
y=xW 
⊤
 +ΔW 
slow
​
 (x)+ΔW 
fast
​
 (x)
self.weight / self.bias：原始线性权重，被冻结不训练

lora_A_slow / lora_B_slow：slow 分支

lora_A_fast / lora_B_fast：fast 分支

scaling = alpha / r

使用 dropout + matmul，兼容 fp16 / bf16

replace_with_ds_lora(...)
遍历模型所有模块：

识别 LLaMA / Qwen / Gemma 等的 decoder layer 类型

对名字包含 target_modules 中任一子串且是 nn.Linear 的层进行替换

对于 layer_idx < num_frozen_layers 的层，不插入 LoRA（完全冻结）

典型调用：

python
复制代码
model = replace_with_ds_lora(
    model,
    target_modules=("q_proj", "v_proj"),
    r=cfg.lora_r,
    alpha=cfg.lora_alpha,
    dropout=cfg.lora_dropout,
    num_frozen_layers=cfg.num_frozen_layers,
)
get_ds_lora_param_groups(...)
遍历模型中所有 DSLoRALinear：

收集 slow 参数 → 一个 param group（lr=lr_slow）

收集 fast 参数 → 一个 param group（lr=lr_fast）

返回：

python
复制代码
optim_groups, slow_params, fast_params
用于：

创建主 optimizer（AdamW）

单独给 slow 分支搞一个 KD 专用 optimizer（在 SLSD 中用）

📁 train_single_task.py
单任务训练脚本（验证 DS-LoRA）：

命令行参数：

bash
复制代码
python train_single_task.py --task T1_general
--task ∈ {T1_general, T2_math, T3_code, T4_tool, T5_safety}

关键流程：

创建 BaseConfig()，根据 cfg.use_toy_data 决定使用：

toy: data/T*_xxx.jsonl

full: data/full/T*_full.jsonl

自动选择一块最空闲 GPU（按剩余显存）

加载 tokenizer & LLaMA 模型：

AutoTokenizer.from_pretrained(cfg.model_name)

AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=float16, device_map="auto")

调用 replace_with_ds_lora(...) 注入双分支 LoRA

使用 InstructionDataset + collate_fn 构建 DataLoader

通过 get_ds_lora_param_groups(...) 建立 slow/fast 不同学习率的 optimizer

标准训练循环：

gradient accumulation

每 logging_steps 写入：

logs/single_<task>_train_loss.jsonl

每个 epoch 结束后：

在当前任务上评估平均 loss → *_eval_loss.jsonl

保存 checkpoint 至 checkpoints/single_<task>_epochX/

📁 train_slsd_seq.py
五任务序列训练脚本（核心 CIT 实验）：

任务顺序：

text
复制代码
T1_general → T2_math → T3_code → T4_tool → T5_safety
配置：

默认 cfg.use_slsd = True

使用 select_device() 选择 GPU

根据 cfg.use_toy_data / cfg.data_paths 选择数据文件

对每个任务 t：

设置 teacher_model = deepcopy(current_model)（上一阶段）

调用 train_one_task_with_slsd(...)：

如果是第一任务：从基座模型 + DS-LoRA 开始

否则：延续上一阶段模型（slow&fast 参数），再注入当前任务训练

若启用 SLSD：

用 teacher_model + 当前任务数据构建 probe buffer

每若干步使用 buffer 计算 KD loss

KD 只更新 slow 分支

保存当前阶段模型：

checkpoints/seq_T1_general/

checkpoints/seq_T2_math/
等等

评估：

在所有“已见任务”上做平均 loss 评估：

例如训练完 T3 后在 T1, T2, T3 上都评估

写入：

logs/seq_eval_loss.jsonl

这是论文中主要用来评估 forgtting + illusory forgetting 的实验脚本。

📁 plot_loss.py
用于可视化训练与评估日志。

示例：

bash
复制代码
python plot_loss.py single T1_general   # 单任务训练 loss 曲线
python plot_loss.py seq                 # 多任务序列的评估 loss 曲线
生成：

logs/single_T1_general_train_loss.png

logs/seq_eval_loss.png

📁 .gitignore（特别说明）
忽略完整数据集：

data/full/

data/raw/

带 _full 的 jsonl

只保留轻量且必要的：

data/T1_general.jsonl ~ data/T5_safety.jsonl（toy 小数据，可选）

data/README.md（如果存在）

这样既能保证仓库轻量，也不会把大型数据集同步到远程。

⚙️ 3. 环境安装
建议使用 conda：

bash
复制代码
conda create -n dslora python=3.10 -y
conda activate dslora
安装 PyTorch（示例：CUDA 11.8）：

bash
复制代码
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
安装其余依赖：

bash
复制代码
pip install transformers datasets tqdm matplotlib accelerate
# 如需量化可额外安装：bitsandbytes
🧪 4. 数据集准备
在服务器上运行：

bash
复制代码
python prepare_datasets.py
脚本会自动生成：

data/full/T1_general_full.jsonl

data/full/T2_math_full.jsonl

data/full/T3_code_full.jsonl

data/full/T4_tool_full.jsonl

data/full/T5_safety_full.jsonl

toy 版本（data/T*_xxx.jsonl）可通过从 full 中抽样获得，用于本地快速调试。

🔥 5. 训练流程
▶ 5.1 单任务训练（验证 DS-LoRA）
示例：在 T1（通用指令任务）上训练：

bash
复制代码
python train_single_task.py --task T1_general
其他任务：

bash
复制代码
python train_single_task.py --task T2_math
python train_single_task.py --task T3_code
python train_single_task.py --task T4_tool
python train_single_task.py --task T5_safety
输出：

训练日志：logs/single_<task>_train_loss.jsonl

评估日志：logs/single_<task>_eval_loss.jsonl

模型：checkpoints/single_<task>_epochX/

▶ 5.2 持续任务序列训练（DS-LoRA + SLSD）
bash
复制代码
python train_slsd_seq.py
训练顺序固定为：T1 → T2 → T3 → T4 → T5

输出：

评估日志：logs/seq_eval_loss.jsonl

阶段模型：

checkpoints/seq_T1_general/

checkpoints/seq_T2_math/

...

checkpoints/seq_T5_safety/

📈 6. 可视化
单任务 loss 曲线：

bash
复制代码
python plot_loss.py single T1_general
# 输出：logs/single_T1_general_train_loss.png
序列任务多任务评估曲线：

bash
复制代码
python plot_loss.py seq
# 输出：logs/seq_eval_loss.png
❗ 7. 常见问题（FAQ）
7.1 Loss 出现 NaN？
使用 float16 时可能出现数值不稳定，建议：

降低学习率（已经设为 lr_slow = 1e-5, lr_fast = 5e-5）

减小 max_seq_len 或 batch size

如仍出现：

在需要时改为 torch_dtype=torch.float32（会更慢但更稳）

7.2 CUDA OOM？
减小：

per_device_batch_size

max_seq_len

增大：

gradient_accumulation_steps

如果显存仍不够：

考虑 4-bit / 8-bit 量化加载 LLaMA

🙏 8. 致谢
本项目基于以下开源工作和工具：

HuggingFace Transformers / Datasets
HuggingFace Transformers / Datasets

PyTorch

LoRA (Hu et al.)

各开放指令数据集（Alpaca, Dolly, GSM8K, CodeAlpaca, HH-RLHF, function-calling 数据等）

以及你的硕士论文：DS-LoRA + SLSD 持续指令微调框架

欢迎引用、扩展与改进本项目。