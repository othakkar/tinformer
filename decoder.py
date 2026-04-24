import jax
import jax.numpy as jnp
from attention import multi_head_attention
from layer_norm import layer_norm
from dataclasses import dataclass

@dataclass
class Attention:
  W_q: jax.Array
  W_k: jax.Array
  W_v: jax.Array
  W_o: jax.Array
  H: int
  ln_scale: jax.Array
  ln_bias: jax.Array

@dataclass
class FFN:
  W1: jax.Array
  b1: jax.Array
  W2: jax.Array
  b2: jax.Array
  ln_scale: jax.Array
  ln_bias: jax.Array

def decoder_block(hidden_states, attn_params, ffn_params, attention_mask=None):
  # hidden_states: (B, T, D_model)

  normed_hidden_states = layer_norm(
      hidden_states, attn_params.ln_scale, attn_params.ln_bias
  )
  attention_out = multi_head_attention(
      normed_hidden_states,
      attn_params.W_q,
      attn_params.W_k,
      attn_params.W_v,
      attn_params.W_o,
      attn_params.H,
      attention_mask,
  )
  residual_after_attention = hidden_states + attention_out

  normed_residual = layer_norm(
      residual_after_attention, ffn_params.ln_scale, ffn_params.ln_bias
  )
  ffn_hidden = jnp.dot(normed_residual, ffn_params.W1) + ffn_params.b1
  ffn_activated = jax.nn.gelu(ffn_hidden)
  ffn_out = jnp.dot(ffn_activated, ffn_params.W2) + ffn_params.b2

  output_states = residual_after_attention + ffn_out
  return output_states  # (B, T, D_model)


if __name__ == "__main__":
  B = 16  # batch size
  T = 128  # sequence length
  D_model = 512  # embedding dimension
  D_k = 32  # dimension of the key and query vectors
  D_v = 32  # dimension of the value vectors
  H = 8  # number of attention heads

  # Random input data
  X = jax.random.normal(jax.random.PRNGKey(0), (B, T, D_model))

  # Random attention weight matrices
  W_q = jax.random.normal(jax.random.PRNGKey(1), (D_model, H * D_k))
  W_k = jax.random.normal(jax.random.PRNGKey(2), (D_model, H * D_k))
  W_v = jax.random.normal(jax.random.PRNGKey(3), (D_model, H * D_v))
  W_o = jax.random.normal(jax.random.PRNGKey(4), (H * D_v, D_model))

  # Random ffn weight matrices
  W1 = jax.random.normal(jax.random.PRNGKey(5), (D_model, 4 * D_model))
  b1 = jax.random.normal(jax.random.PRNGKey(6), (4 * D_model,))
  W2 = jax.random.normal(jax.random.PRNGKey(7), (4 * D_model, D_model))
  b2 = jax.random.normal(jax.random.PRNGKey(8), (D_model,))

  # Random LayerNorm parameter initialization
  ln1_scale = jnp.ones((D_model,))
  ln1_bias = jnp.zeros((D_model,))
  ln2_scale = jnp.ones((D_model,))
  ln2_bias = jnp.zeros((D_model,))

  # Mask
  causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))  # (T, T), True = keep

  attn_params = Attention(W_q, W_k, W_v, W_o, H, ln1_scale, ln1_bias)
  ffn_params = FFN(W1, b1, W2, b2, ln2_scale, ln2_bias)

  decoder_output = decoder_block(
      X, attn_params=attn_params, ffn_params=ffn_params, attention_mask=causal_mask
  )
  print("Decoder output shape: ", decoder_output.shape)