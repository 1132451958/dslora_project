# models/ds_lora.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DSLoRALinear(nn.Module):
    """
    把原始 nn.Linear 替换成这个：
    y = x W^T + slow(x) + fast(x)
    其中 W 冻结，slow & fast 是低秩 LoRA 分支
    """

    def __init__(self, base_layer: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        # 冻结的原始权重
        self.weight = base_layer.weight
        self.bias = base_layer.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

        dtype = base_layer.weight.dtype
        device = base_layer.weight.device

        # slow branch
        self.lora_A_slow = nn.Parameter(
            torch.zeros(r, self.in_features, dtype=dtype, device=device)
        )
        self.lora_B_slow = nn.Parameter(
            torch.zeros(self.out_features, r, dtype=dtype, device=device)
        )

        # fast branch
        self.lora_A_fast = nn.Parameter(
            torch.zeros(r, self.in_features, dtype=dtype, device=device)
        )
        self.lora_B_fast = nn.Parameter(
            torch.zeros(self.out_features, r, dtype=dtype, device=device)
        )

        self.reset_parameters()

    def reset_parameters(self):
        # A 用正态，B 用零初始化比较常见
        nn.init.kaiming_uniform_(self.lora_A_slow, a=5**0.5)
        nn.init.zeros_(self.lora_B_slow)
        nn.init.kaiming_uniform_(self.lora_A_fast, a=5**0.5)
        nn.init.zeros_(self.lora_B_fast)

    def forward(
        self,
        x: torch.Tensor,
        lora_slow: bool = True,
        lora_fast: bool = True,
    ) -> torch.Tensor:
        # x 的 dtype 是 float16，LoRA 权重已经跟它一致了
        result = F.linear(x, self.weight, self.bias)

        if not (lora_slow or lora_fast):
            return result

        x_ = self.dropout(x)

        if lora_slow:
            delta_slow = (x_ @ self.lora_A_slow.t()) @ self.lora_B_slow.t()
            result = result + self.scaling * delta_slow

        if lora_fast:
            delta_fast = (x_ @ self.lora_A_fast.t()) @ self.lora_B_fast.t()
            result = result + self.scaling * delta_fast

        return result


def replace_with_ds_lora(
    model: nn.Module,
    target_modules,
    r: int,
    alpha: int,
    dropout: float,
    num_frozen_layers: int = 0,
):
    """
    - target_modules: 模块名里包含这些子串的 Linear 会被替换（如 "q_proj", "v_proj"）
    - num_frozen_layers: 冻结底部若干层（只在更高层加 DS-LoRA）
    对 TinyLlama/LLaMA 系模型适用
    """
    try:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    except Exception:
        # 如果以后你换别的架构（如 Qwen），这里需要按新模型结构改一下
        LlamaDecoderLayer = None

    layer_idx = -1

    for module in model.modules():
        if LlamaDecoderLayer is not None and isinstance(module, LlamaDecoderLayer):
            layer_idx += 1

        for name, child in list(module.named_children()):
            # 只对名字里包含 target_modules 中任意一个关键词的 Linear 层做替换
            if not any(t in name for t in target_modules):
                continue
            if not isinstance(child, nn.Linear):
                continue

            # 如果设置了冻结底层层数，则只对更高层进行 LoRA 替换
            if layer_idx >= 0 and layer_idx < num_frozen_layers:
                continue

            ds_layer = DSLoRALinear(child, r=r, alpha=alpha, dropout=dropout)
            setattr(module, name, ds_layer)

    return model


def get_ds_lora_param_groups(
    model: nn.Module,
    lr_slow: float,
    lr_fast: float,
    weight_decay: float,
):
    """
    返回 optimizer 的 param groups，
    并把 slow / fast 分别放到不同学习率组里
    """
    slow_params = []
    fast_params = []

    for module in model.modules():
        if isinstance(module, DSLoRALinear):
            slow_params.extend([module.lora_A_slow, module.lora_B_slow])
            fast_params.extend([module.lora_A_fast, module.lora_B_fast])

    optim_groups = [
        {"params": slow_params, "lr": lr_slow, "weight_decay": weight_decay},
        {"params": fast_params, "lr": lr_fast, "weight_decay": weight_decay},
    ]

    return optim_groups, slow_params, fast_params
