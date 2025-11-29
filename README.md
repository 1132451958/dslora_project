DS-LoRA & SLSD: Continual Instruction Tuning Framework 说明文档
1. 论文目标与问题设定
1.1 论文主题

论文题目（暂定）

DS-LoRA + SLSD: Stability-aware Efficient Continual Instruction Tuning Framework

研究目标：
在 持续指令微调（Continual Instruction Tuning, CIT） 场景下，让一个 LLaMA-2-7B 指令模型按任务序列依次学习：

T1 General → T2 Math → T3 Code → T4 Tool-Calling → T5 Safety

同时尽量避免两类遗忘：

知识遗忘（Catastrophic Forgetting）

旧任务上的客观性能明显下降（如准确率、loss 变差）。

虚假遗忘（Illusory Forgetting）

模型在旧任务上还 “会做题”，但：

输出风格变了（口吻、格式不一致）

alignment 变差

safety 降低（更容易给出不安全回答）

论文提出两个核心方法：

Dual-Speed LoRA（DS-LoRA）：结构层面缓解遗忘

Stability-aware Lightweight Self-Distillation（SLSD）：蒸馏层面缓解“虚假遗忘”

2. 方法概览
2.1 DS-LoRA

核心想法：在 LoRA 上做“快慢双分支”，兼顾 稳定性（slow） 和 可塑性（fast）。

冻结底层 Transformer，大致包含：

self-attn / FFN 原始权重

embedding / LM head 等
→ 保持 alignment / safety / 文本风格不漂移。

对每个需要适配的 Linear 层，替换为：

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


Slow 分支：

所有任务共享

lr_slow 很小（例如 5e-6 / 1e-5）

负责长期稳定、累积“通用风格”和知识

Fast 分支：

每个任务独立 LoRA

lr_fast 较大（例如 2e-5 / 5e-5）

负责快速适应当前任务特性

训练时：

Fast 分支：主 supervised loss + 大学习率

Slow 分支：主 supervised loss + SLSD 蒸馏约束（小学习率）

2.2 SLSD（Stability-aware Lightweight Self-Distillation）

目标：在 CIT 中抑制 “虚假遗忘”（风格漂移），不需要保存大量旧数据。

训练第 t 个任务时：

使用上一阶段模型 
𝜃
𝑡
−
1
θ
t−1
	​

 作为 teacher；

在当前任务数据上构建一个 probe buffer：

采样少量样本（例如 100–500 条）

运行 teacher 前向，计算最后一个 token 的 预测熵；

只保留熵较低（teacher 自信）的样本，并缓存其 logits；

在训练当前任务时：

主 supervised loss 正常更新 slow + fast；

每隔若干步，从 probe buffer 抽一个样本，对当前模型 logits 与 teacher logits 做 KL 蒸馏；

只用 slow 分支的 optimizer 更新（fast 分支不被 KD“抹平”）。

优势：

不需要完整保留旧任务数据，只用轻量的 probe buffer；

蒸馏只约束 slow 分支，可以稳定风格、对齐性和安全性，而不削弱 fast 的可塑性。

3. 实验设置与数据
3.1 模型与训练设定

基座模型：LLaMA-2-7B（HF，本地路径在 BaseConfig.model_name 中指定）

训练类型：Causal LM（instruction → output）

数据格式（统一 JSONL）：

{"instruction": "...", "input": "...", "output": "..."}


在 InstructionDataset 中会被转换为 LLaMA 指令格式：

[INST] instruction \n input [/INST] output </s>


labels 只监督 output 区域（prompt 部分 label = -100）。

3.2 任务与数据来源

数据构建脚本：prepare_datasets.py → 生成 data/full/T?_*_full.jsonl，之后由 split_datasets.py 拆分。

T1_general：Alpaca + Dolly 类通用指令

T2_math：GSM8K(train) 风格分步解题

T3_code：CodeAlpaca-20k 风格代码生成/解释

T4_tool：Glaive function-calling（OpenAI style 函数调用），解析成 JSON 字符串输出

T5_safety：HH-RLHF / 安全对话数据

3.3 数据拆分与清洗

脚本：split_datasets.py（最新版，带清洗 + 每任务最大 10K 样本）

对每个任务：

从 data/full/{task}_full.jsonl 读入

清洗逻辑：

必须包含 instruction / output 字段

转成字符串后 strip() 不能为空

使用 LLaMA tokenizer 估算完整序列 token 数量，超过 BaseConfig.max_seq_len 的样本丢弃

若清洗后样本数 > 10,000：

随机采样到 最多 10,000 条（保证训练成本可控）

再按 80/10/10 拆分为：

*_train.jsonl

*_val.jsonl

*_test.jsonl

从 train 中再采样 tiny 集：

*_tiny.jsonl，大小约 min(1000, 0.1 * train)

拆分结果保存在：

data/split/
  T1_general_train.jsonl
  T1_general_val.jsonl
  T1_general_test.jsonl
  T1_general_tiny.jsonl
  ...
  T5_safety_*.jsonl

4. 仓库目录结构与关键文件说明

根目录：dslora_project/

dslora_project/
├── configs/
│   └── base_config.py
├── data/
│   ├── full/           # prepare_datasets.py 生成的 full 数据（未拆分）
│   └── split/          # split_datasets.py 生成的 train/val/test/tiny
├── logs/
│   ├── seq_eval_loss.jsonl
│   ├── seq_eval_loss.png
│   ├── single_*_train_loss.jsonl
│   ├── seq_*_train_loss.jsonl  # 各方法各任务的训练 loss
│   └── *.png                   # 绘图输出
├── models/
│   ├── ds_lora.py
│   └── lora_simple.py
├── checkpoints/
│   └── ...                    # 各阶段保存的模型与 tokenizer
├── prepare_datasets.py
├── split_datasets.py
├── utils_data.py
├── train_single_task.py
├── train_slsd_seq.py
├── train_seq_baselines.py
└── plot_loss.py

4.1 configs/base_config.py

核心配置 dataclass，集中管理：

模型路径：

model_name = "pretrained_models/llama2-7b-hf"


LoRA / DS-LoRA 超参：

lora_r, lora_alpha, lora_dropout

lora_target_modules = ("q_proj", "v_proj")

num_frozen_layers = 16

lr_slow, lr_fast

weight_decay

数据路径 data_paths：

以 task 为 key，包含 train/val/test/tiny/full/toy 的文件路径

数据 split 选择：

use_toy_data：是否使用 toy 小数据

train_split, eval_split, test_split

通用训练参数：

max_seq_len

per_device_batch_size

gradient_accumulation_steps

num_epochs

save_dir

Replay / EWC 相关超参（baseline 用）：

replay_buffer_size, replay_lambda

ewc_lambda 等

SLSD 相关：

use_slsd

kd_lambda

probe_size_per_task

entropy_threshold

4.2 utils_data.py

InstructionDataset：

读取 jsonl

拼接 LLaMA [INST] ... [/INST] prompt

max_length = cfg.max_seq_len

labels 中 prompt 部分全部置为 -100，只监督 output

collate_fn：

做 padding，返回 input_ids, attention_mask, labels

4.3 models/ds_lora.py

DS-LoRA 核心实现：

将指定的 Linear 模块替换为带 slow / fast 双 LoRA 的模块

提供：

replace_with_ds_lora(model, ...)

get_ds_lora_param_groups(model, lr_slow, lr_fast, weight_decay)

返回 slow 参数组、fast 参数组，用于分别设置学习率

4.4 models/lora_simple.py

单分支 LoRA 实现，用于 baseline：

replace_with_lora

mark_only_lora_as_trainable

get_lora_param_groups(model, lr, weight_decay)

4.5 数据相关脚本

prepare_datasets.py

从原始公开数据集（Alpaca, GSM8K, CodeAlpaca, Glaive FC, HH-RLHF 等）构建统一格式 data/full/T*_full.jsonl。

split_datasets.py（新版，带清洗 + 10K cap）

详见上文第 3.3。

4.6 训练脚本
4.6.1 train_single_task.py

功能：在单任务上训练 DS-LoRA（不考虑持续学习）

用法：

python train_single_task.py --task T1_general
# or T2_math / T3_code / T4_tool / T5_safety


输出：

logs/single_T?_train_loss.jsonl

checkpoints/single_task/...

角色：验证 DS-LoRA 在单任务场景下的效果，作为 CIT 的参考上界。

4.6.2 train_slsd_seq.py（主方法：DS-LoRA + SLSD）

功能：在任务序列 T1→T5 上进行 CIT，采用 DS-LoRA + SLSD。

主要流程：

第一个任务：

加载 LLaMA-7B

注入 DS-LoRA

冻结非 LoRA 参数

每个任务 t：

使用前一阶段模型作为 teacher（深拷贝 & eval）

调用 build_probe_buffer：

从当前任务数据采样，teacher 前向

计算最后一个 token 的熵，熵低的样本加入 buffer，缓存 logits

主训练循环：

supervised loss：更新 slow + fast

每隔若干步对 buffer 中样本做 KD：

用 kd_loss_from_buffer 计算 KL

只用 slow LoRA 的 optimizer 更新（fast 只受 supervised loss）

训练日志：logs/seq_T?_train_loss.jsonl

保存阶段模型：checkpoints/seq_T?_*

对已见任务做 eval，写入 logs/seq_eval_loss.jsonl

最新版本中已加入：

NaN / Inf loss & gradient 检查

labels 全为 -100 的 batch 直接跳过

eval 空数据检测

4.6.3 train_seq_baselines.py（基线：Seq-LoRA / Replay / EWC）

命令行参数 --method：

seq_lora

replay

ewc

使用单分支 LoRA（models/lora_simple.py），多任务顺序训练：

seq_lora：纯顺序微调

replay：加入 replay buffer，从旧任务混入样本

ewc：对 LoRA 参数估计 Fisher，对重要参数加 EWC 正则

训练日志：

logs/seq_seq_lora_T?_train_loss.jsonl

logs/seq_replay_T?_train_loss.jsonl

logs/seq_ewc_T?_train_loss.jsonl

评估：

统一写入 logs/seq_eval_loss.jsonl（带 method 字段）

最新版本同样带 NaN/Inf 检查、全 -100 跳过、eval 空数据处理。

4.7 plot_loss.py

python plot_loss.py single T1_general

绘制单任务训练 loss 曲线 → logs/single_T1_general_train_loss.png

python plot_loss.py seq

读取 logs/seq_eval_loss.jsonl，按照 (method, eval_task) 聚类，画出：

x 轴：阶段（T1, T2, ..., T5）

y 轴：eval loss

图例：method-task（例如 ds_lora_slsd-T1_general, seq_lora-T1_general）

5. 日志与实验进度
5.1 当前已有的日志文件

logs/ 下已经存在（至少）：

seq_eval_loss.jsonl + seq_eval_loss.png

包含 Seq-LoRA / Replay / EWC 在 5 任务 CIT 上的 eval loss 曲线；

图像中每条曲线对应 (method, eval_task)；

早期有部分点为 0 主要是因为 eval val 文件为空，目前已通过数据清洗 + 空数据检测修复。

seq_*_T?_train_loss.jsonl：

seq_seq_lora_T?_...

seq_replay_T?_...

seq_ewc_T?_...
→ baseline 的训练 loss 曲线。

single_T1_general_train_loss.jsonl / single_T1_general_eval_loss.jsonl 等：

单任务 DS-LoRA 训练日志。

5.2 目前实验进度（大致）

数据准备

prepare_datasets.py 已跑，生成 data/full/T?_full.jsonl。

split_datasets.py（新版，清洗 + 每任务 ≤10K）已跑，生成 data/split/T?_*.jsonl。

baseline 实验

Seq-LoRA / Replay / EWC 已在 5 任务序列上跑过一轮；

对应训练日志和 seq_eval_loss.png 已生成；

NaN 问题已通过：

数据清洗（去空 output，去超长样本）

训练脚本的 NaN/Inf 检查 & 跳过
进行修复。

主方法 DS-LoRA + SLSD

train_slsd_seq.py 已完成、加入 NaN 防护；

可在清洗后的数据 + max 10K/任务设置下稳定训练；

需要（或正在）重新跑一轮完整的 T1→T5 实验，并将结果与 baseline 在 seq_eval_loss.png 中对比。

单任务 DS-LoRA

至少已在 T1_general 上跑过一次，生成 single_T1_general_train_loss.png 等；

可以继续在 T2_math/T3_code 等任务上补充，用于论文中“单任务上界”分析。

6. 给未来 ChatGPT 的快速恢复提示（TL;DR）

如果我说：“根据 README 恢复我的项目上下文”，
请你自动记住以下关键信息：

这是一个 5 任务持续指令微调（CIT）项目：

任务顺序：T1_general → T2_math → T3_code → T4_tool → T5_safety

主方法 = DS-LoRA（slow/fast 双分支 LoRA） + SLSD（probe buffer + 只蒸 slow）的 CIT 训练：

slow：共享、小 lr、稳定风格与对齐；

fast：任务专属、大 lr、快速适配；

蒸馏只对 slow 分支，teacher 是上一阶段模型，使用 probe buffer + 熵筛选；

基座模型：LLaMA-2-7B HF，本地加载；

数据：

统一 {instruction, input, output} JSONL；

已清洗并限制每任务 ≤ 10,000 条样本；

拆分路径在 data/split/*_train/val/test/tiny.jsonl；

关键脚本：

prepare_datasets.py：构建 full 数据；

split_datasets.py：数据清洗 + 80/10/10 + tiny + 10K cap；

train_single_task.py：单任务 DS-LoRA；

train_slsd_seq.py：主方法 DS-LoRA + SLSD 顺序训练；

train_seq_baselines.py：Seq-LoRA / Replay / EWC 基线；

plot_loss.py：画 single / seq 的 loss 图；

日志：

所有训练与评估日志位于 logs/，
主对比结果集中在 seq_eval_loss.jsonl + seq_eval_loss.png。

只要记住这些，你就可以在任何时候从我的 repo 结构中快速定位对应文件、继续帮我改代码 / 调参 / 写论文实验分析。