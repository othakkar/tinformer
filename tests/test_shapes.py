"""Verify output shapes of each component."""
import jax
import jax.numpy as jnp
from config import TinformerConfig
from attention import MultiHeadAttention
from ffn import FFN  # assuming similar interface
from decoder import DecoderBlock
from tinformer import Tinformer

config = TinformerConfig()
B, T, D = config.B, config.T, config.D_model

def test_mha_output_shape():
    mha = MultiHeadAttention(D, config.D_k, config.D_v, config.H)
    X = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    out, (K, V) = mha(X, mask=mask)
    assert out.shape == (B, T, D)
    assert K.shape == (B, config.H, T, config.D_k)
    assert V.shape == (B, config.H, T, config.D_v)

def test_tinformer_output_shape():
    model = Tinformer(config)
    token_ids = jax.random.randint(jax.random.PRNGKey(0), (B, T), 0, config.vocab_size)
    logits, caches = model.forward(token_ids)
    assert logits.shape == (B, T, config.vocab_size)
    assert len(caches) == config.N
