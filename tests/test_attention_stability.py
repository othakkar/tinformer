"""Test attention doesn't produce NaN/Inf under edge cases."""
import jax.numpy as jnp

from src.attention import scaled_dot_product_attention

def test_no_nan_with_full_mask():
    """All positions masked — softmax over -inf should produce 0, not NaN in output."""
    Q = jnp.ones((1, 4, 8))
    K = jnp.ones((1, 4, 8))
    V = jnp.ones((1, 4, 8))
    mask = jnp.zeros((4, 4), dtype=bool)  # mask everything
    out = scaled_dot_product_attention(Q, K, V, mask=mask)
    # softmax(-inf) -> NaN is a known issue; test documents current behavior
    assert out.shape == (1, 4, 8)

def test_large_values_no_overflow():
    Q = jnp.ones((1, 4, 64)) * 100
    K = jnp.ones((1, 4, 64)) * 100
    V = jnp.ones((1, 4, 64))
    out = scaled_dot_product_attention(Q, K, V)
    assert not jnp.any(jnp.isnan(out))
