# models/ds_lora.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================
# DS-LoRA Linear（支持 slow + fast 双分支）
# ======================================================
class DSLoRALinear(nn.Module):
    """
    用于替换 nn.Linear:
      y = x W^T + ΔW_slow(x) + ΔW_fast(x)

    W 完全冻结，slow/fast 为可训练 LoRA 分支。
    """

    def __init__(self, base_layer: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        # --- 冻结原始权重 ---
        self.weight = base_layer.weight
        self.bias = base_layer.bias

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # --- LoRA 参数 ---
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

        dtype = base_layer.weight.dtype
        device = base_layer.weight.device

        # slow branch
        self.lora_A_slow = nn.Parameter(torch.zeros(r, self.in_features, dtype=dtype, device=device))
        self.lora_B_slow = nn.Parameter(torch.zeros(self.out_features, r, dtype=dtype, device=device))

        # fast branch
        self.lora_A_fast = nn.Parameter(torch.zeros(r, self.in_features, dtype=dtype, device=device))
        self.lora_B_fast = nn.Parameter(torch.zeros(self.out_features, r, dtype=dtype, device=device))

        self.reset_parameters()

    def reset_parameters(self):
        # slow 与 fast 的初始化策略一致，更稳定
        nn.init.kaiming_uniform_(self.lora_A_slow, a=5**0.5)
        nn.init.zeros_(self.lora_B_slow)
        nn.init.kaiming_uniform_(self.lora_A_fast, a=5**0.5)
        nn.init.zeros_(self.lora_B_fast)

    # --------------------------------------------------
    # 前向计算（支持 fp16/bf16 安全 matmul）
    # --------------------------------------------------
    def forward(self, x, lora_slow=True, lora_fast=True):
        # 原始线性输出（只读）
        result = F.linear(x, self.weight, self.bias)

        if not (lora_slow or lora_fast):
            return result

        x_ = self.dropout(x)

        # --- slow branch ---
        if lora_slow:
            # matmul 更安全（避免 fp16 overflow）
            delta_slow = torch.matmul(x_, self.lora_A_slow.t())
            delta_slow = torch.matmul(delta_slow, self.lora_B_slow.t())
            result = result + self.scaling * delta_slow

        # --- fast branch ---
        if lora_fast:
            delta_fast = torch.matmul(x_, self.lora_A_fast.t())
            delta_fast = torch.matmul(delta_fast, self.lora_B_fast.t())
            result = result + self.scaling * delta_fast

        return result


# ======================================================
# 替换 Linear 为 DS-LoRA
# ======================================================
def replace_with_ds_lora(
    model: nn.Module,
    target_modules,
    r: int,
    alpha: int,
    dropout: float,
    num_frozen_layers: int = 0,
):
    """
    target_modules = ("q_proj", "v_proj") 这样的关键词列表
    num_frozen_layers = 冻结底部多少层（不插 LoRA）
      对 LLaMA（32 层），num_frozen_layers=16 代表冻结前 16 层

    支持：
        - LlamaDecoderLayer（LLaMA）
        - QWenBlock（Qwen/Qwen2）
        - GemmaDecoderLayer（Gemma）
    """

    # 多架构支持
    try:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    except:
        LlamaDecoderLayer = None

    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer as QwenBlock
    except:
        QwenBlock = None

    try:
        from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
    except:
        GemmaDecoderLayer = None

    # 当前层号
    layer_idx = -1

    def is_transformer_layer(module):
        """判断是否是一个 decoder layer（用于层计数）"""
        if LlamaDecoderLayer is not None and isinstance(module, LlamaDecoderLayer):
            return True
        if QwenBlock is not None and isinstance(module, QwenBlock):
            return True
        if GemmaDecoderLayer is not None and isinstance(module, GemmaDecoderLayer):
            return True
        return False

    # 开始替换
    for module in model.modules():
        # 层计数（每遇到一个 transformer block，则层号+1）
        if is_transformer_layer(module):
            layer_idx += 1

        # 遍历 submodules
        for name, child in list(module.named_children()):
            # 目标模块名称匹配，例如 "q_proj"
            if not any(t in name for t in target_modules):
                continue

            # 不是 Linear 的跳过
            if not isinstance(child, nn.Linear):
                continue

            # 冻结层：不添加 LoRA
            if 0 <= layer_idx < num_frozen_layers:
                continue

            # 替换成 DS-LoRA
            ds_layer = DSLoRALinear(child, r=r, alpha=alpha, dropout=dropout)
            setattr(module, name, ds_layer)

    return model


# ======================================================
# 获取 slow & fast 参数组（用于 optimizer）
# ======================================================
def get_ds_lora_param_groups(model, lr_slow, lr_fast, weight_decay):
    slow_params = []
    fast_params = []

    for module in model.modules():
        if isinstance(module, DSLoRALinear):
            slow_params.append(module.lora_A_slow)
            slow_params.append(module.lora_B_slow)
            fast_params.append(module.lora_A_fast)
            fast_params.append(module.lora_B_fast)

    optim_groups = [
        {"params": slow_params, "lr": lr_slow, "weight_decay": weight_decay},
        {"params": fast_params, "lr": lr_fast, "weight_decay": weight_decay},
    ]

    return optim_groups, slow_params, fast_params
