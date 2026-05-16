"""Verify output shapes of each component."""
import jax
import jax.numpy as jnp

from src.config import TinformerConfig
from src.attention import MultiHeadAttention
from src.tinformer import Tinformer

config = TinformerConfig()
B, T, D = config.B, config.T, config.D_model

def test_mha_output_shape():
    mha = MultiHeadAttention(D, config.D_k, config.D_v, config.H)
    X = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    out = mha(X, mask=mask)
    assert out[0].shape == (B, T, D)

def test_tinformer_output_shape():
    model = Tinformer(config)
    token_ids = jax.random.randint(jax.random.PRNGKey(0), (B, T), 0, config.vocab_size)
    logits, _ = model.forward(token_ids)
    assert logits.shape == (B, T, config.vocab_size)
