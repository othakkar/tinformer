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

def multi_head_attention(X, W_q, W_k, W_v, W_o, H, mask=None):
  # X: (B, T, D_model)
  # W_q: (D_model, H * D_k)
  # W_k: (D_model, H * D_k)
  # W_v: (D_model, H * D_v)
  # W_o: (H * D_v, D_model)
  
  B, T, D_model = X.shape
  D_k = W_q.shape[-1] // H
  D_v = W_v.shape[-1] // H

  Q = jnp.dot(X, W_q)  # (B, T, H * D_k)
  Q = Q.reshape(B, T, H, D_k).transpose(0, 2, 1, 3)  # (B, H, T, D_k)
  K = jnp.dot(X, W_k)  # (B, T, H * D_k)
  K = K.reshape(B, T, H, D_k).transpose(0, 2, 1, 3)  # (B, H, T, D_k)
  V = jnp.dot(X, W_v)  # (B, T, H * D_v)
  V = V.reshape(B, T, H, D_v).transpose(0, 2, 1, 3)  # (B, H, T, D_v)

  Z = scaled_dot_product_attention(Q, K, V, mask=mask)  # (B, H, T, D_v)

  Z = Z.transpose(0, 2, 1, 3).reshape(B, T, H * D_v)  # (B, T, H * D_v)

  output = jnp.dot(Z, W_o)  # (B, T, D_model)
  return output
  

B = 16  # batch size
T = 128  # sequence length
D_model = 512  # embedding dimension
D_k = 32  # dimension of the key and query vectors
D_v = 32  # dimension of the value vectors
H = 8  # number of attention heads

# Random input data and weight matrices
X = jax.random.normal(jax.random.PRNGKey(0), (B, T, D_model))
W_q = jax.random.normal(jax.random.PRNGKey(1), (D_model, H * D_k))
W_k = jax.random.normal(jax.random.PRNGKey(2), (D_model, H * D_k))
W_v = jax.random.normal(jax.random.PRNGKey(3), (D_model, H * D_v))
W_o = jax.random.normal(jax.random.PRNGKey(4), (H * D_v, D_model))

causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))  # (T, T), True = keep

output = multi_head_attention(X, W_q, W_k, W_v, W_o, H, mask=causal_mask)
print(output.shape)  # (B, T, D_model)