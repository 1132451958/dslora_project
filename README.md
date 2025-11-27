📘 DS-LoRA & SLSD: Continual Instruction Tuning Framework

（硕士论文官方实现仓库）

本项目实现了硕士论文提出的：

Dual-Speed LoRA（DS-LoRA）

Stability-aware Lightweight Self-Distillation（SLSD）

用于解决 持续指令微调（Continual Instruction Tuning, CIT） 中的：

虚假遗忘（Illusory Forgetting）：输出风格 / 对齐性 / 安全性漂移

知识遗忘（Forgetting）：旧任务性能下降

本仓库包含完整实现：数据集构建、训练脚本、DS-LoRA + SLSD 模型、baselines、可视化工具等，支持从头复现论文实验。

🧠 0. 快速上下文恢复（给未来 ChatGPT 的提示）

如果用户说：
“请根据 README 恢复我的项目上下文”

请自动记住以下内容：

项目是五任务的持续指令微调（CIT）：
General → Math → Code → Tool-Calling → Safety

主方法 = DS-LoRA（慢/快双分支 LoRA） + SLSD（轻量蒸馏）

slow 分支：共享、稳定、小 lr

fast 分支：任务专属、大 lr

底层 Transformer 冻结，避免风格漂移

SLSD：只蒸 slow，缓解虚假遗忘

基座模型：LLaMA-2-7B HF（本地加载）

完整的数据集已构建在：data/full/*_full.jsonl

T4 使用最新的 glaive function calling (openai-style) 数据构建

训练脚本：

train_single_task.py：单任务 DS-LoRA

train_slsd_seq.py：五任务顺序训练（DS-LoRA + SLSD）

train_seq_baselines.py：Seq-LoRA / Replay / EWC baseline

LoRA/路径/冻结层等主要超参全部在：
configs/base_config.py

DS-LoRA 全部代码在：
models/ds_lora.py

数据模板遵循 LLaMA [INST] ... [/INST] 格式，labels 只监督 output

所有训练、评估、绘图日志在：logs/

🚀 1. 论文方法概述
1.1 Dual-Speed LoRA（DS-LoRA）

目标：同时保持模型的 稳定性（抗遗忘） 与 可塑性（快速学新任务），避免风格漂移（虚假遗忘）。

✓ 冻结底层 Transformer（如前 16 层）

保持基座模型的：

alignment（对齐）

safety（安全性）

style（文本风格）

tokenizer & decoding behavior

✓ 线性层替换为 Slow + Fast 双 LoRA 分支

对每个 Linear：

𝑊
=
𝑊
0
+
Δ
𝑊
slow
+
Δ
𝑊
fast
W=W
0
	​

+ΔW
slow
	​

+ΔW
fast
	​


Slow（共享）

learning rate 小

负责长期稳定性

所有任务共享

Fast（任务专属）

learning rate 大

负责快速学习当前任务

每个任务独立

✓ 优化器拆分

slow 使用 lr_slow（如 1e-5）

fast 使用 lr_fast（如 5e-5）

1.2 SLSD（Stability-aware Lightweight Self-Distillation）

目标：防止“虚假遗忘”（风格漂移）。

流程：

Probe Buffer（100–500样本）：来自当前任务

使用上一阶段模型 teacher（θ_{t-1}）前向一次生成 logits

按熵筛选最能代表“旧风格”的样本

只对 slow 分支做 KD 蒸馏，fast 不蒸馏

优势：

不需要保存大量旧数据

不需要 teacher 多次前向

蒸馏只作用于 slow 分支，避免抹平 fast 分支的可塑性

📂 2. 项目结构
dslora_project/
│
├── configs/
│   └── base_config.py               # 全局配置（LoRA超参/路径/冻结层等）
│
├── data/
│   ├── T*_*.jsonl                   # toy 小数据
│   └── full/
│       ├── T1_general_full.jsonl
│       ├── T2_math_full.jsonl
│       ├── T3_code_full.jsonl
│       ├── T4_tool_full.jsonl       # Glaive FC 解析，自定义格式
│       └── T5_safety_full.jsonl
│
├── logs/
│   ├── single_*_train_loss.jsonl
│   ├── seq_eval_loss.jsonl
│   └── *.png                        # 绘图输出
│
├── models/
│   └── ds_lora.py                   # DS-LoRA 核心实现（slow/fast 双分支）
│   └── lora_simple.py               # baseline 用的单分支 LoRA
│
├── prepare_datasets.py              # 构建五个任务 full 数据
├── split_datasets.py                # 80/10/10 + tiny 拆分
│
├── utils_data.py                    # LLaMA instrcut 格式 + mask labels
│
├── train_single_task.py             # 单任务 DS-LoRA
├── train_slsd_seq.py                # 主方法：DS-LoRA + SLSD
├── train_seq_baselines.py           # Seq-LoRA / Replay / EWC
│
└── plot_loss.py                     # 训练/评估可视化

📊 3. 数据集准备 (已完成)

运行：

python prepare_datasets.py


生成：

Task	数据来源	full 大小
T1	Alpaca + Dolly	~67k
T2	GSM8K(train)	~7.4k
T3	CodeAlpaca-20k	~20k
T4	Glaive Function-Calling (openai-style)	~10~20k（解析后）
T5	HH-RLHF	~100k

统一格式：

{
  "instruction": "...",
  "input": "",
  "output": "..."   // 对 T4 是函数调用 JSON 字符串
}


然后运行：

python split_datasets.py


得到：

data/split/T?_train.jsonl
data/split/T?_val.jsonl
data/split/T?_test.jsonl
data/split/T?_tiny.jsonl

🔥 4. 训练方式
4.1 单任务训练（验证 DS-LoRA）
python train_single_task.py --task T1_general


其它任务：

T2_math
T3_code
T4_tool
T5_safety


结果：

logs/single_*/train_loss.jsonl

checkpoints/single_task/…

4.2 完整五任务序列训练（主方法：DS-LoRA + SLSD）
python train_slsd_seq.py


顺序：

T1 → T2 → T3 → T4 → T5


每阶段：

保存模型（checkpoints/seq_*/）

评估所有已见任务

写入日志：logs/seq_eval_loss.jsonl

4.3 Baselines（单 LoRA 分支）

Seq-LoRA：

python train_seq_baselines.py --method seq_lora


Replay：

python train_seq_baselines.py --method replay


EWC：

python train_seq_baselines.py --method ewc

📈 5. 可视化
单任务 loss：
python plot_loss.py single T1_general

序列任务 forgetting 曲线：
python plot_loss.py seq


生成：

logs/single_*.png

logs/seq_eval_loss.png

❗ 6. 常见问题 (FAQ)
Loss 出现 NaN？

减小学习率

减小 max_seq_len

使用 bf16（默认已是）

对数学任务可开启 gradient_checkpointing

显存不足？

batch_size=1（默认）

增大 gradient_accumulation_steps

使用 bitsandbytes 量化加载模型

T4 风格与其他任务不同，会不会影响虚假遗忘？

不会。
虚假遗忘来自风格漂移，而五任务本身风格非常不同。
格式统一不会破坏异质性实验，反而更干净。

🙏 致谢

本项目基于以下开源工作：

LLaMA / HuggingFace Transformers

LoRA (Hu et al.)

Alpaca / Dolly / GSM8K / CodeAlpaca

HH-RLHF（Anthropic）

Glaive function-calling dataset

PyTorch / Datasets

以及你的硕士论文：
DS-LoRA + SLSD: Stability-aware Efficient Continual Instruction Tuning Framework