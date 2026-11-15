"""
lib/models/sd_encoder/lora.py -- LoRA cho U-Net SD (chep lai inject_trainable_lora cua cloneofsimo/lora, MIT; 2026-11-15).

Giong lora_diffusion (ban DPMesh goi: inject_trainable_lora(unet, r=64), dropout_p 0, scale 1):
  - dich: moi nn.Linear nam trong CrossAttention (to_q, to_k, to_v, to_out.0) va GEGLU (proj)
    (DEFAULT_TARGET_REPLACE = {'CrossAttention', 'Attention', 'GEGLU'});
  - LoraInjectedLinear: y = linear(x) + dropout(lora_up(lora_down(x))) * scale,
    lora_down ~ N(0, 1 / r), lora_up = 0 -> luc dau giong het mo hinh goc.
GOI SAU KHI da nap trong so SD va dong bang U-Net (trong so goc chuyen vao .linear, giu requires_grad = False).
"""
import torch.nn as nn

from .blocks import CrossAttention, GEGLU


class LoraInjectedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, r=4, dropout_p=0.0, scale=1.0):
        super().__init__()
        if r > min(in_features, out_features):
            raise ValueError(f'LoRA rank {r} phai <= min({in_features}, {out_features})')
        self.r = r
        self.linear = nn.Linear(in_features, out_features, bias)
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = scale
        self.selector = nn.Identity()
        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.linear(x) + self.dropout(self.lora_up(self.selector(self.lora_down(x)))) * self.scale


def inject_trainable_lora(model, r=4, dropout_p=0.0, scale=1.0, targets=(CrossAttention, GEGLU)):
    """Thay nn.Linear trong cac module dich bang LoraInjectedLinear (giu nguyen trong so goc).
    Tra (list tham so LoRA, list ten Linear da thay)."""
    found, seen = [], set()
    for anc in model.modules():
        if not isinstance(anc, targets):
            continue
        for anc_name, parent in anc.named_modules():
            if isinstance(parent, LoraInjectedLinear):
                continue
            for name, child in parent.named_children():
                if type(child) is nn.Linear and id(child) not in seen:      # dung type: khong boc lai .linear
                    seen.add(id(child))
                    found.append((parent, name, child))
    params, names = [], []
    for parent, name, child in found:
        new = LoraInjectedLinear(child.in_features, child.out_features, child.bias is not None,
                                 r=r, dropout_p=dropout_p, scale=scale)
        new.linear.weight = child.weight
        if child.bias is not None:
            new.linear.bias = child.bias
        new.lora_down.to(device=child.weight.device, dtype=child.weight.dtype)
        new.lora_up.to(device=child.weight.device, dtype=child.weight.dtype)
        parent._modules[name] = new
        params.extend(list(new.lora_down.parameters()) + list(new.lora_up.parameters()))
        names.append(name)
    return params, names
