# configs/base_config.py
from dataclasses import dataclass

@dataclass
class BaseConfig:
    # ====== 模型 & LoRA 设置 ======
    # 本地调试建议先用 tiny 模型：TinyLlama-1.1B 或 facebook/opt-350m
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules = ("q_proj", "v_proj")  # LLaMA 系模型常用

    # DS-LoRA 相关
    num_frozen_layers: int = 16      # 对于 32 层 LLaMA，可以冻前 16 层；小模型调试先设 0-2
    lr_slow: float = 1e-5
    lr_fast: float = 5e-5
    weight_decay: float = 0.01

    # SLSD 相关
    use_slsd: bool = False
    kd_lambda: float = 0.5
    probe_size_per_task: int = 500
    entropy_threshold: float = 1.5

    # ====== 训练相关 ======
    train_file: str = "data/toy.jsonl"
    max_seq_len: int = 512
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_epochs: int = 1
    logging_steps: int = 10
    save_dir: str = "checkpoints"

    # ====== 设备 ======
    device: str = "cuda"
