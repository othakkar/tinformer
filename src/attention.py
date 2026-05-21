import math

import jax
import jax.numpy as jnp

from src.config import TinformerConfig

def scaled_dot_product_attention(Q, K, V, mask=None):
  D_k = Q.shape[-1]
  scores = (
    jnp.matmul(Q, jnp.swapaxes(K, -1, -2)) / jnp.sqrt(D_k)
  )

  if mask is not None:
    scores = jnp.where(mask, scores, -jnp.inf)

  attn = jax.nn.softmax(scores, axis=-1)
  return jnp.matmul(attn, V)

class MultiHeadAttention:
  def __init__(self, D_model, D_k, D_v, H, num_kv_heads, key=jax.random.PRNGKey(0)):
    self.H = H
    self.D_k = D_k
    self.D_v = D_v
    self.num_kv_heads = num_kv_heads

    assert H % num_kv_heads == 0

    keys = jax.random.split(key, 4)
    self.W_q = jax.random.normal(keys[0], (D_model, H * D_k)) / math.sqrt(D_model)
    self.W_k = jax.random.normal(keys[1], (D_model, num_kv_heads * D_k)) / math.sqrt(D_model)
    self.W_v = jax.random.normal(keys[2], (D_model, num_kv_heads * D_v)) / math.sqrt(D_model)
    self.W_o = jax.random.normal(keys[3], (H * D_v, D_model)) / math.sqrt(H * D_v)
  
  def __call__(self, X, mask=None, kv_cache=None):
    B, T, _ = X.shape  # T=1 when kv_cache is not None
    Q = jnp.dot(X, self.W_q) # (B, T, H * D_k)
    Q = Q.reshape(B, T, self.H, self.D_k).transpose(0, 2, 1, 3) # (B, H, T, D_k)
    K = jnp.dot(X, self.W_k) # (B, T, num_kv_heads * D_k)
    K = K.reshape(B, T, self.num_kv_heads, self.D_k).transpose(0, 2, 1, 3) # (B, num_kv_heads, T, D_k)
    V = jnp.dot(X, self.W_v) # (B, T, num_kv_heads * D_v)
    V = V.reshape(B, T, self.num_kv_heads, self.D_v).transpose(0, 2, 1, 3) # (B, num_kv_heads, T, D_v)

    if kv_cache is not None:
      cached_K, cached_V = kv_cache
      K = jnp.concatenate([cached_K, K], axis=2)
      V = jnp.concatenate([cached_V, V], axis=2)

    new_kv_cache = (K, V)

    K = jnp.repeat(K, self.H // self.num_kv_heads, axis=1) # (B, H, T, D_k)
    V = jnp.repeat(V, self.H // self.num_kv_heads, axis=1) # (B, H, T, D_v)

    Z = scaled_dot_product_attention(Q, K, V, mask=mask) # (B, H, T, D_v)
    Z = Z.transpose(0, 2, 1, 3).reshape(B, T, self.H * self.D_v) # (B, T, H * D_v)
    return jnp.dot(Z, self.W_o), new_kv_cache # (B, T, D_model)
  
if __name__ == "__main__":
  # MHA
  config = TinformerConfig()
  mha = MultiHeadAttention(config.D_model, config.D_k, config.D_v, config.H, config.num_kv_heads)
  X = jax.random.normal(jax.random.PRNGKey(0), (config.B, config.T, config.D_model))
  mask = jnp.tril(jnp.ones((config.T, config.T), dtype=bool)) # (T, T)
  print("MultiHeadAttention output shape: ", mha(X, mask=mask)[0].shape)

  # MQA
  config = TinformerConfig(num_kv_heads=1)
  mqa = MultiHeadAttention(config.D_model, config.D_k, config.D_v, config.H, config.num_kv_heads)
  X = jax.random.normal(jax.random.PRNGKey(0), (config.B, config.T, config.D_model))
  mask = jnp.tril(jnp.ones((config.T, config.T), dtype=bool)) # (T, T)
  print("MultiHeadAttention output shape: ", mqa(X, mask=mask)[0].shape)

  # GQA
  config = TinformerConfig(num_kv_heads=4)
  gqa = MultiHeadAttention(config.D_model, config.D_k, config.D_v, config.H, config.num_kv_heads)
  X = jax.random.normal(jax.random.PRNGKey(0), (config.B, config.T, config.D_model))
  mask = jnp.tril(jnp.ones((config.T, config.T), dtype=bool)) # (T, T)
  print("MultiHeadAttention output shape: ", gqa(X, mask=mask)[0].shape)
