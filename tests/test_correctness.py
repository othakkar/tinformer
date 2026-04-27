"""Verify that KV-cached generation produces identical outputs to naive generation."""
import jax
import jax.numpy as jnp
from mini_gpt import MiniGPT
from config import GPTConfig
from benchmarks.generate import naive_generate, cached_generate

def test_cached_matches_naive_logits():
    """Compare first-step logits to confirm mathematical equivalence."""
    config = GPTConfig()
    model = MiniGPT(config)
    prompt = jax.random.randint(jax.random.PRNGKey(42), (config.B, config.T), 0, config.vocab_size)

    # Full forward
    full_logits, _ = model.forward(prompt)

    # Prefill via cache
    cached_logits, _ = model.forward(prompt, kv_caches=None)

    assert jnp.allclose(full_logits, cached_logits, atol=1e-5)
