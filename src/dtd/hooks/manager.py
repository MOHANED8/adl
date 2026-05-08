from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class HookSpec:
    """
    Controls what internal tensors get captured.

    - module_globs: patterns matched against module qualified names, e.g. ["layer1.*", "blocks.*.attn"]
    - capture_inputs: store (first) module input tensors
    - capture_outputs: store module outputs (activations)
    - capture_output_grads: store gradients of module outputs (requires backward call)
    - capture_attention: heuristic capture for attention modules:
        stores module output if it looks like an attention map (B,H,T,T) OR if output is tuple (x, attn)
    """

    module_globs: tuple[str, ...] = ("*",)
    capture_inputs: bool = False
    capture_outputs: bool = True
    capture_output_grads: bool = True
    capture_attention: bool = True
    detach: bool = True
    cpu: bool = True
    max_per_module: int = 1


def _to_store(x: Any, *, detach: bool, cpu: bool):
    if isinstance(x, torch.Tensor):
        if detach:
            x = x.detach()
        if cpu:
            x = x.to("cpu")
        return x
    if isinstance(x, (tuple, list)):
        return type(x)(_to_store(v, detach=detach, cpu=cpu) for v in x)
    if isinstance(x, dict):
        return {k: _to_store(v, detach=detach, cpu=cpu) for k, v in x.items()}
    return x


class HookManager:
    def __init__(self, model: nn.Module, spec: Optional[HookSpec] = None):
        self.model = model
        self.spec = spec or HookSpec()
        self.handles: list[Any] = []
        self.clear()

    def clear(self):
        self.activations: dict[str, list[Any]] = {}
        self.gradients: dict[str, list[Any]] = {}
        self.attention: dict[str, list[Any]] = {}
        self.embeddings: dict[str, list[Any]] = {}

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def _match(self, name: str) -> bool:
        return any(fnmatch(name, pat) for pat in self.spec.module_globs)

    def attach(self):
        self.remove()

        for name, module in self.model.named_modules():
            if name == "":
                continue
            if not self._match(name):
                continue

            def _fwd_hook(mod, inp, out, *, _name=name):
                # Inputs
                if self.spec.capture_inputs and inp:
                    self._push(self.embeddings, f"{_name}:in", inp[0])

                # Outputs (activations)
                if self.spec.capture_outputs:
                    self._push(self.activations, _name, out)

                # Attention heuristic
                if self.spec.capture_attention:
                    attn = None
                    if hasattr(mod, "_last_attn"):
                        try:
                            attn_obj = getattr(mod, "_last_attn")
                            if isinstance(attn_obj, torch.Tensor) and attn_obj.ndim == 4:
                                attn = attn_obj
                        except Exception:
                            attn = None
                    if attn is None:
                        attn = self._extract_attention(_name, out)
                    if attn is not None:
                        self._push(self.attention, _name, attn)

                # Gradients of outputs
                if self.spec.capture_output_grads and isinstance(out, torch.Tensor) and out.requires_grad:
                    out.register_hook(lambda g, _n=_name: self._push(self.gradients, _n, g))

            self.handles.append(module.register_forward_hook(_fwd_hook))

        return self

    def _push(self, store: dict[str, list[Any]], key: str, value: Any):
        value = _to_store(value, detach=self.spec.detach, cpu=self.spec.cpu)
        lst = store.setdefault(key, [])
        if self.spec.max_per_module > 0 and len(lst) >= self.spec.max_per_module:
            lst.pop(0)
        lst.append(value)

    @staticmethod
    def _extract_attention(module_name: str, out: Any) -> Optional[torch.Tensor]:
        # Common pattern: (x, attn) or {"attn": ...}
        if isinstance(out, (tuple, list)) and len(out) == 2 and isinstance(out[1], torch.Tensor):
            attn = out[1]
            if attn.ndim == 4:
                return attn
        if isinstance(out, dict) and "attn" in out and isinstance(out["attn"], torch.Tensor):
            attn = out["attn"]
            if attn.ndim == 4:
                return attn
        # Only treat raw 4D tensor as attention if module name indicates it.
        if isinstance(out, torch.Tensor) and out.ndim == 4:
            mn = module_name.lower()
            if "attn" in mn or "attention" in mn:
                return out
        return None

