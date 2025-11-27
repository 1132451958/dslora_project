# configs/base_config.py
import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

# =======================
# 项目根目录 & 本地模型路径
# =======================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 本地 LLaMA-2-7B HF 模型目录
LOCAL_LLAMA_PATH = os.path.join(PROJECT_ROOT, "pretrained_models", "llama2-7b-hf")


@dataclass
class BaseConfig:
    # =======================
    # 模型名称（本地路径）
    # =======================
    model_name: str = LOCAL_LLAMA_PATH

    # =======================
    # LoRA / DS-LoRA 配置
    # =======================
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # 对 LLaMA 系模型：q_proj / v_proj 是常见注入位置
    lora_target_modules: Tuple[str, ...] = ("q_proj", "v_proj")

    # 冻结前 N 层（LLaMA-2-7B 共 32 层）
    num_frozen_layers: int = 16

    # DS-LoRA 学习率（单任务 / DS-LoRA 使用）
    lr_slow: float = 5e-6
    lr_fast: float = 2e-5
    weight_decay: float = 0.01

    # =======================
    # SLSD 相关
    # =======================
    use_slsd: bool = False
    kd_lambda: float = 0.5
    probe_size_per_task: int = 500
    entropy_threshold: float = 1.5

    # =======================
    # 数据集路径配置
    # =======================
    # 是否使用 toy 数据（快速测试）
    use_toy_data: bool = False

    # 训练/验证/测试 使用哪种 split key
    # ⬇⬇⬇ 这里改成 train，默认用完整训练集
    train_split: str = "train"   # "tiny" 用于快速调试，正式实验改为 "train"
    eval_split: str = "val"
    test_split: str = "test"

    # 5 个任务的数据路径（toy / full / split）
    data_paths: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: {
            "T1_general": {
                "toy": "data/T1_general.jsonl",
                "full": "data/full/T1_general_full.jsonl",
                "train": "data/split/T1_general_train.jsonl",
                "val": "data/split/T1_general_val.jsonl",
                "test": "data/split/T1_general_test.jsonl",
                "tiny": "data/split/T1_general_tiny.jsonl",
            },
            "T2_math": {
                "toy": "data/T2_math.jsonl",
                "full": "data/full/T2_math_full.jsonl",
                "train": "data/split/T2_math_train.jsonl",
                "val": "data/split/T2_math_val.jsonl",
                "test": "data/split/T2_math_test.jsonl",
                "tiny": "data/split/T2_math_tiny.jsonl",
            },
            "T3_code": {
                "toy": "data/T3_code.jsonl",
                "full": "data/full/T3_code_full.jsonl",
                "train": "data/split/T3_code_train.jsonl",
                "val": "data/split/T3_code_val.jsonl",
                "test": "data/split/T3_code_test.jsonl",
                "tiny": "data/split/T3_code_tiny.jsonl",
            },
            "T4_tool": {
                "toy": "data/T4_tool.jsonl",
                "full": "data/full/T4_tool_full.jsonl",
                "train": "data/split/T4_tool_train.jsonl",
                "val": "data/split/T4_tool_val.jsonl",   # 记得后面补一些样本，否则 eval loss 会是 0
                "test": "data/split/T4_tool_test.jsonl",
                "tiny": "data/split/T4_tool_tiny.jsonl",
            },
            "T5_safety": {
                "toy": "data/T5_safety.jsonl",
                "full": "data/full/T5_safety_full.jsonl",
                "train": "data/split/T5_safety_train.jsonl",
                "val": "data/split/T5_safety_val.jsonl",
                "test": "data/split/T5_safety_test.jsonl",
                "tiny": "data/split/T5_safety_tiny.jsonl",
            },
        }
    )

    # 默认训练文件（会被 train_*.py 中覆盖）
    train_file: str = "data/T1_general.jsonl"

    # =======================
    # 训练参数
    # =======================
    max_seq_len: int = 1024
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_epochs: int = 1
    logging_steps: int = 10
    save_dir: str = "checkpoints"

    # 梯度裁剪（0 或 None 代表不裁剪）
    clip_grad: float = 1.0

    # =======================
    # Replay / EWC 相关超参（给 baselines 用）
    # =======================
    replay_buffer_size: int = 2048
    replay_every: int = 10
    replay_lambda: float = 1.0

    ewc_lambda: float = 0.4
    ewc_fisher_batches: int = 200  # 近似 Fisher 时使用的 batch 数（toy 可以改小）

    # =======================
    # 设备
    # =======================
    device: str = "cuda"   # 没 GPU 可以改为 "cpu"
