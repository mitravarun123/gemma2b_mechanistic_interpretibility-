"""Offline integration tests against real HuggingFace Gemma attention modules."""
import pytest
import torch
from transformers import GemmaConfig, GemmaForCausalLM, Gemma2Config, Gemma2ForCausalLM
from src.model.loader import architecture, attention_projections
from src.model.hooks import capture_heads, intervene


@pytest.fixture(params=['gemma', 'gemma2'])
def model(request):
    torch.manual_seed(42)
    config_type, model_type = (GemmaConfig, GemmaForCausalLM) if request.param == 'gemma' else (Gemma2Config, Gemma2ForCausalLM)
    config = config_type(vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=32, attn_implementation='eager', sliding_window=4)
    return model_type(config).eval()


@torch.no_grad()
def test_capture_self_patch_and_zero(model):
    tokens = torch.tensor([[2, 5, 7, 11, 13, 17]])
    with capture_heads(model) as states:
        baseline = model(tokens).logits
    assert states[0].shape == (1, 6, 4, 8)
    with intervene(model, {(0, 1): states[0][:, :, 1, :]}):
        patched = model(tokens).logits
    torch.testing.assert_close(patched, baseline, rtol=0, atol=0)
    with intervene(model, {(0, 1): None}):
        zeroed = model(tokens).logits
    assert not torch.allclose(zeroed, baseline, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(model(tokens).logits, baseline, rtol=0, atol=0)


@torch.no_grad()
def test_target_position_mean_and_cleanup(model):
    tokens = torch.tensor([[2, 5, 7, 11]])
    with capture_heads(model) as states:
        baseline = model(tokens).logits
    with intervene(model, {(1, 2): states[1][:, -1:, 2, :]}, positions=[3]):
        torch.testing.assert_close(model(tokens).logits, baseline, rtol=0, atol=0)
    with intervene(model, {(0, 1): torch.zeros(8)}):
        assert torch.isfinite(model(tokens).logits).all()
    with pytest.raises(RuntimeError):
        with intervene(model, {(0, 0): None}):
            raise RuntimeError('intentional failure')
    assert all(len(module._forward_pre_hooks) == 0 for module in attention_projections(model))
    with pytest.raises(ValueError):
        with intervene(model, {(0, 4): None}):
            pass
    assert architecture(model.config).n_kv_heads == 1


@torch.no_grad()
def test_full_state_patch_restores_clean(model):
    clean = torch.tensor([[2, 5, 7, 11]])
    corrupted = torch.tensor([[2, 7, 5, 11]])
    with capture_heads(model) as states:
        clean_logits = model(clean).logits[:, -1]
    # Replacing only attention outputs does not replace the residual stream.
    # Full clean state is tested via final hidden-state replacement instead.
    layer = model.model.layers[-1]
    cached = {}
    def save(module, args, output):
        cached['state'] = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
    handle = layer.register_forward_hook(save)
    model(clean)
    handle.remove()
    def restore(module, args, output):
        return (cached['state'], *output[1:]) if isinstance(output, tuple) else cached['state']
    handle = layer.register_forward_hook(restore)
    restored = model(corrupted).logits[:, -1]
    handle.remove()
    torch.testing.assert_close(restored, clean_logits)
    corrupted_logits = model(corrupted).logits[:, -1]
    delta = clean_logits - corrupted_logits
    valid = delta.abs() > 1e-6
    assert valid.any()
    torch.testing.assert_close(((restored-corrupted_logits)/delta)[valid], torch.ones_like(delta[valid]))
    torch.testing.assert_close(((corrupted_logits-corrupted_logits)/delta)[valid], torch.zeros_like(delta[valid]))
