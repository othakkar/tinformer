"""Tests for MQA and GQA configurations."""
import pytest
import jax
import jax.numpy as jnp

from src.config import TinformerConfig
from src.attention import MultiHeadAttention
from src.tinformer import Tinformer


B, T, D_model, D_k, D_v, H = 2, 16, 64, 16, 16, 8


class TestMQAConfig:
    """Multi-Query Attention: num_kv_heads=1."""

    def test_config_mqa(self):
        config = TinformerConfig(num_kv_heads=1)
        assert config.num_kv_heads == 1

    def test_mha_output_shape_mqa(self):
        mha = MultiHeadAttention(D_model, D_k, D_v, H, num_kv_heads=1)
        X = jax.random.normal(jax.random.PRNGKey(0), (B, T, D_model))
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        out, kv_cache = mha(X, mask=mask)
        assert out.shape == (B, T, D_model)

    def test_kv_cache_shape_mqa(self):
        mha = MultiHeadAttention(D_model, D_k, D_v, H, num_kv_heads=1)
        X = jax.random.normal(jax.random.PRNGKey(0), (B, T, D_model))
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        _, (K, V) = mha(X, mask=mask)
        # KV cache should have num_kv_heads=1 heads
        assert K.shape == (B, 1, T, D_k)
        assert V.shape == (B, 1, T, D_v)

    def test_tinformer_with_mqa(self):
        config = TinformerConfig(
            B=B, T=T, D_model=D_model, D_k=D_k, D_v=D_v,
            H=H, vocab_size=50, N=2, max_seq_len=64, num_kv_heads=1
        )
        model = Tinformer(config)
        token_ids = jax.random.randint(jax.random.PRNGKey(0), (B, T), 0, config.vocab_size)
        logits, caches = model.forward(token_ids)
        assert logits.shape == (B, T, config.vocab_size)
        # Each layer's cache should have 1 KV head
        assert caches[0][0].shape == (B, 1, T, D_k)


class TestGQAConfig:
    """Grouped-Query Attention: 1 < num_kv_heads < H."""

    def test_config_gqa(self):
        config = TinformerConfig(num_kv_heads=4)
        assert config.num_kv_heads == 4

    def test_mha_output_shape_gqa(self):
        mha = MultiHeadAttention(D_model, D_k, D_v, H, num_kv_heads=4)
        X = jax.random.normal(jax.random.PRNGKey(0), (B, T, D_model))
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        out, kv_cache = mha(X, mask=mask)
        assert out.shape == (B, T, D_model)

    def test_kv_cache_shape_gqa(self):
        mha = MultiHeadAttention(D_model, D_k, D_v, H, num_kv_heads=4)
        X = jax.random.normal(jax.random.PRNGKey(0), (B, T, D_model))
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        _, (K, V) = mha(X, mask=mask)
        assert K.shape == (B, 4, T, D_k)
        assert V.shape == (B, 4, T, D_v)

    def test_tinformer_with_gqa(self):
        config = TinformerConfig(
            B=B, T=T, D_model=D_model, D_k=D_k, D_v=D_v,
            H=H, vocab_size=50, N=2, max_seq_len=64, num_kv_heads=4
        )
        model = Tinformer(config)
        token_ids = jax.random.randint(jax.random.PRNGKey(0), (B, T), 0, config.vocab_size)
        logits, caches = model.forward(token_ids)
        assert logits.shape == (B, T, config.vocab_size)
        assert caches[0][0].shape == (B, 4, T, D_k)


class TestMHAConfig:
    """Standard MHA: num_kv_heads == H."""

    def test_kv_cache_shape_mha(self):
        mha = MultiHeadAttention(D_model, D_k, D_v, H, num_kv_heads=H)
        X = jax.random.normal(jax.random.PRNGKey(0), (B, T, D_model))
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        _, (K, V) = mha(X, mask=mask)
        assert K.shape == (B, H, T, D_k)
        assert V.shape == (B, H, T, D_v)


class TestInvalidConfig:
    """num_kv_heads must divide H evenly."""

    def test_invalid_num_kv_heads(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(D_model, D_k, D_v, H, num_kv_heads=3)


class TestOutputEquivalence:
    """MQA/GQA should produce same output shape and be deterministic."""

    def test_mqa_gqa_mha_same_output_shape(self):
        X = jax.random.normal(jax.random.PRNGKey(0), (B, T, D_model))
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))

        for num_kv_heads in [1, 2, 4, 8]:
            mha = MultiHeadAttention(D_model, D_k, D_v, H, num_kv_heads=num_kv_heads)
            out, _ = mha(X, mask=mask)
            assert out.shape == (B, T, D_model), f"Failed for num_kv_heads={num_kv_heads}"
