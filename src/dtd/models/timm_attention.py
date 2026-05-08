from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def enable_timm_attention_capture(model: nn.Module) -> nn.Module:
    """
    Best-effort patch for timm ViT/DeiT attention modules to expose attention maps.

    After patching, attention modules will store `module._last_attn` shaped (B, H, T, T)
    on each forward pass.
    """
    for _, m in model.named_modules():
        if m.__class__.__name__ != "Attention":
            continue
        if not (hasattr(m, "qkv") and hasattr(m, "proj") and hasattr(m, "num_heads")):
            continue

        # Avoid double patching
        if getattr(m, "_dtd_patched", False):
            continue

        def _forward(x: torch.Tensor, *, _m=m) -> torch.Tensor:
            B, N, C = x.shape
            qkv = _m.qkv(x).reshape(B, N, 3, _m.num_heads, C // _m.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * getattr(_m, "scale", 1.0)
            attn = attn.softmax(dim=-1)
            if hasattr(_m, "attn_drop"):
                attn = _m.attn_drop(attn)
            _m._last_attn = attn

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = _m.proj(x)
            if hasattr(_m, "proj_drop"):
                x = _m.proj_drop(x)
            return x

        m.forward = _forward  # type: ignore[method-assign]
        m._dtd_patched = True

    return model

