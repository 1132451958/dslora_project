# models/lora_simple.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    简单单分支 LoRA：
        y = x W^T + ΔW(x)
    其中 W 冻结，LoRA 分支可训练。
    """

    def __init__(self, base_layer: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        # 冻结原始权重
        self.weight = base_layer.weight
        self.bias = base_layer.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # LoRA 参数
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

        dtype = base_layer.weight.dtype
        device = base_layer.weight.device

        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=dtype, device=device))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = F.linear(x, self.weight, self.bias)

        x_ = self.dropout(x)
        delta = torch.matmul(x_, self.lora_A.t())
        delta = torch.matmul(delta, self.lora_B.t())
        result = result + self.scaling * delta

        return result


def replace_with_lora(
    model: nn.Module,
    target_modules,
    r: int,
    alpha: int,
    dropout: float,
    num_frozen_layers: int = 0,
):
    """
    与 ds_lora.replace_with_ds_lora 类似，但只插入单分支 LoRA。
    用于 Seq-LoRA / Replay / EWC baseline。
    """

    try:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    except Exception:
        LlamaDecoderLayer = None

    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer as QwenBlock
    except Exception:
        QwenBlock = None

    try:
        from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
    except Exception:
        GemmaDecoderLayer = None

    def is_transformer_layer(module):
        if LlamaDecoderLayer is not None and isinstance(module, LlamaDecoderLayer):
            return True
        if QwenBlock is not None and isinstance(module, QwenBlock):
            return True
        if GemmaDecoderLayer is not None and isinstance(module, GemmaDecoderLayer):
            return True
        return False

    layer_idx = -1

    for module in model.modules():
        if is_transformer_layer(module):
            layer_idx += 1

        for name, child in list(module.named_children()):
            if not any(t in name for t in target_modules):
                continue
            if not isinstance(child, nn.Linear):
                continue

            # 冻结底部若干层：不插 LoRA
            if 0 <= layer_idx < num_frozen_layers:
                continue

            lora_layer = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
            setattr(module, name, lora_layer)

    return model


def mark_only_lora_as_trainable(model: nn.Module):
    """
    非常关键：
    - 先将所有参数 requires_grad=False
    - 再仅打开 LoRALinear 的 A/B 参数

    这样反向传播图和显存都只围绕 LoRA 分支，大幅节省内存。
    """
    # 全部先关掉
    for _, p in model.named_parameters():
        p.requires_grad = False

    # 只打开 LoRA 参数
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True


def get_lora_param_groups(model, lr: float, weight_decay: float):
    """
    返回一个 optimizer param group，包含所有 LoRA 参数。
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            lora_params.append(module.lora_A)
            lora_params.append(module.lora_B)

    optim_groups = [{"params": lora_params, "lr": lr, "weight_decay": weight_decay}]
    return optim_groups
