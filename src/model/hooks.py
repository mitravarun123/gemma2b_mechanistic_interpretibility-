"""Head interventions at the input to attention output projections."""
from contextlib import contextmanager
from collections.abc import Iterator, Sequence
import torch
from .loader import architecture, attention_projections


@contextmanager
def capture_heads(model: torch.nn.Module, layers: Sequence[int] | None = None, to_cpu: bool = False) -> Iterator[dict[int, torch.Tensor]]:
    """Capture detached [batch, sequence, query heads, head_dim] activations."""
    info = architecture(model.config)
    modules = attention_projections(model)
    selected = list(range(info.n_layers)) if layers is None else list(layers)
    if any(layer < 0 or layer >= info.n_layers for layer in selected):
        raise ValueError("Layer index out of range")
    captured = {}
    handles = []
    try:
        for layer in selected:
            def hook(module, args, layer=layer):
                state = args[0].reshape(*args[0].shape[:2], info.n_heads, info.head_dim).detach()
                captured[layer] = state.cpu().clone() if to_cpu else state.clone()
            handles.append(modules[layer].register_forward_pre_hook(hook))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def intervene(model: torch.nn.Module, replacements: dict[tuple[int, int], torch.Tensor | None], positions: Sequence[int] | None = None) -> Iterator[None]:
    """Replace head slices with cached states or reference means; None zero-ablates.

    Replacement tensors must broadcast to [batch, selected positions, head_dim].
    Hooks always clone the input, and are removed even if model execution fails.
    """
    info = architecture(model.config)
    modules = attention_projections(model)
    for layer, head in replacements:
        if not 0 <= layer < info.n_layers or not 0 <= head < info.n_heads:
            raise ValueError("Layer or head index out of range")
    handles = []
    try:
        for layer in sorted({layer for layer, _ in replacements}):
            def hook(module, args, layer=layer):
                original = args[0]
                state = original.reshape(*original.shape[:2], info.n_heads, info.head_dim).clone()
                index = slice(None) if positions is None else list(positions)
                for (target_layer, head), replacement in replacements.items():
                    if target_layer == layer:
                        target = state[:, index, head, :]
                        value = torch.zeros_like(target) if replacement is None else replacement.to(device=target.device, dtype=target.dtype)
                        state[:, index, head, :] = torch.broadcast_to(value, target.shape)
                return (state.reshape_as(original), *args[1:])
            handles.append(modules[layer].register_forward_pre_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()
