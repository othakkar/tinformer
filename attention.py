import jax
import jax.numpy as jnp

def scaled_dot_product_attention(Q, K, V, mask=None):
  # Q: (..., seq_len_q, d_k)
  # K: (..., seq_len_k, d_k)
  # V: (..., seq_len_k, d_v)
  # mask (optional): broadcastable to (..., seq_len_q, seq_len_k) with 1/True = keep, 0/False = mask
  # return: (..., seq_len_q, d_v)

  d_k = Q.shape[-1]
  scores = jnp.matmul(Q, jnp.swapaxes(K, -1, -2)) / jnp.sqrt(d_k)

  if mask is not None:
    scores = jnp.where(mask, scores, -jnp.inf)

  attn = jax.nn.softmax(scores, axis=-1)
  return jnp.matmul(attn, V)

class MultiHeadAttention:
  def __init__(self, D_model, D_k, D_v, H, key=jax.random.PRNGKey(0)):
    self.H = H
    self.D_k = D_k
    self.D_v = D_v

    keys = jax.random.split(key, 4)
    self.W_q = jax.random.normal(keys[0], (D_model, H * D_k))
    self.W_k = jax.random.normal(keys[1], (D_model, H * D_k))
    self.W_v = jax.random.normal(keys[2], (D_model, H * D_v))
    self.W_o = jax.random.normal(keys[3], (H * D_v, D_model))
  
  def __call__(self, X, mask=None):
    B, T, D_model = X.shape

    Q = jnp.dot(X, self.W_q) # (B, T, H * D_k)
    Q = Q.reshape(B, T, self.H, self.D_k).transpose(0, 2, 1, 3) # (B, H, T, D_k)
    K = jnp.dot(X, self.W_k) # (B, T, H * D_k)
    K = K.reshape(B, T, self.H, self.D_k).transpose(0, 2, 1, 3) # (B, H, T, D_k)
    V = jnp.dot(X, self.W_v) # (B, T, H * D_v)
    V = V.reshape(B, T, self.H, self.D_v).transpose(0, 2, 1, 3) # (B, H, T, D_v)
    Z = scaled_dot_product_attention(Q, K, V, mask=mask) # (B, H, T, D_v)
    Z = Z.transpose(0, 2, 1, 3).reshape(B, T, self.H * self.D_v) # (B, T, H * D_v)
    return jnp.dot(Z, self.W_o) # (B, T, D_model)
  
if __name__ == "__main__":
  from config import GPTConfig
  config = GPTConfig()
  mha = MultiHeadAttention(config.D_model, config.D_k, config.D_v, config.H)
  X = jax.random.normal(jax.random.PRNGKey(0), (config.B, config.T, config.D_model))
  mask = jnp.tril(jnp.ones((config.T, config.T), dtype=bool)) # (T, T)
  print("MultiHeadAttention output shape: ", mha(X, mask=mask).shape)
