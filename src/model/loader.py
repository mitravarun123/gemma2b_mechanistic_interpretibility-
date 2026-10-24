"""HuggingFace Gemma loading and architecture discovery."""
import logging
import os
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Architecture:
    """Dimensions and attention behaviors read from the model configuration."""
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    sliding_window: int | None
    attention_softcap: float | None
    final_softcap: float | None


def architecture(config) -> Architecture:
    """Resolve Gemma text dimensions, including explicit Gemma head dimensions."""
    config = getattr(config, "text_config", config)
    return Architecture(config.num_hidden_layers, config.num_attention_heads,
        getattr(config, "num_key_value_heads", config.num_attention_heads),
        getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
        getattr(config, "sliding_window", None),
        getattr(config, "attn_logit_softcapping", None),
        getattr(config, "final_logit_softcapping", None))


def resolve_device(name: str) -> torch.device:
    """Choose CUDA, MPS or CPU unless a device is explicitly configured."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(config: dict):
    """Load eager attention and a tokenizer using HF_TOKEN for gated access."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    settings = config["model"]
    dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    if settings["dtype"] not in dtypes:
        raise ValueError("model.dtype must be float32 or bfloat16")
    kwargs = {"token": os.environ.get("HF_TOKEN"), "revision": settings.get("revision", "main")}
    tokenizer = AutoTokenizer.from_pretrained(settings["name"], **kwargs)
    model = AutoModelForCausalLM.from_pretrained(settings["name"], attn_implementation="eager", torch_dtype=dtypes[settings["dtype"]], **kwargs)
    model.to(resolve_device(settings.get("device", "auto"))).eval()
    info = architecture(model.config)
    logging.info("Architecture: %s; shared KV=%s; layer types=%s", info, info.n_kv_heads < info.n_heads, getattr(model.config, "layer_types", None))
    attention_projections(model)  # Validate hook compatibility before expensive work.
    return model, tokenizer, info


def attention_projections(model) -> list:
    """Discover decoder o_proj modules in layer order, including nested text models."""
    projections = [(name, module) for name, module in model.named_modules() if name.endswith(".self_attn.o_proj")]
    expected = architecture(model.config)
    if len(projections) != expected.n_layers:
        raise ValueError(f"Expected {expected.n_layers} attention projections; found {len(projections)}")
    modules = [module for _, module in projections]
    for module in modules:
        if module.in_features != expected.n_heads * expected.head_dim:
            raise ValueError("o_proj input width does not match query-head dimensions")
    return modules
