"""Verify causal masking: changing future tokens doesn't affect past logits."""
import jax
import jax.numpy as jnp
from mini_gpt import MiniGPT
from config import GPTConfig

def test_causal_no_leakage():
    config = GPTConfig()
    model = MiniGPT(config)
    
    tokens_a = jax.random.randint(jax.random.PRNGKey(0), (1, 10), 0, config.vocab_size)
    tokens_b = tokens_a.at[:, 5:].set(0)  # Change future tokens

    logits_a, _ = model.forward(tokens_a)
    logits_b, _ = model.forward(tokens_b)

    # Logits for positions 0-4 should be identical
    assert jnp.allclose(logits_a[:, :5, :], logits_b[:, :5, :], atol=1e-5)
