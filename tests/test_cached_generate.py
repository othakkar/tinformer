"""Verify that cached generate produces the same tokens as naive generate."""
import jax
import jax.numpy as jnp

from src.config import TinformerConfig
from src.tinformer import Tinformer
from generate import cached_generate, naive_generate


def test_cached_matches_naive():
    config = TinformerConfig(B=2, T=16, D_model=64, D_k=16, D_v=16, H=4, vocab_size=50, N=2, max_seq_len=64)
    model = Tinformer(config)
    num_tokens_to_generate = 8

    input_prompt_token_ids = jax.random.randint(
        jax.random.PRNGKey(42), (config.B, config.T), minval=0, maxval=config.vocab_size
    )

    naive_tokens = naive_generate(model, input_prompt_token_ids, num_tokens_to_generate)
    cached_tokens = cached_generate(model, input_prompt_token_ids, num_tokens_to_generate)

    assert naive_tokens.shape == cached_tokens.shape
    assert jnp.array_equal(naive_tokens, cached_tokens), (
        f"Mismatch between naive and cached generate:\n"
        f"  naive:  {naive_tokens}\n"
        f"  cached: {cached_tokens}"
    )
