import jax
import jax.numpy as jnp

from src.attention import MultiHeadAttention
from src.layernorm import LayerNorm
from src.ffn import FFN
from src.config import TinformerConfig

class DecoderBlock:
  def __init__(
      self, D_model, D_k, D_v, H, key=jax.random.PRNGKey(0)
  ):
    keys = jax.random.split(key, 2)
    self.attn = MultiHeadAttention(
        D_model, D_k, D_v, H, key=keys[0]
    )
    self.ln_attn = LayerNorm(D_model)
    self.ln_ffn = LayerNorm(D_model)

    self.ffn = FFN(D_model, key=keys[1])
  
  def __call__(self, hidden_states, attention_mask=None, kv_cache=None):
    normed_hidden_states = self.ln_attn(hidden_states)
    attention_out, new_kv_cache = self.attn(
        normed_hidden_states, mask=attention_mask, kv_cache=kv_cache
    )
    residual_after_attention = attention_out + hidden_states
    normed_residual = self.ln_ffn(residual_after_attention)
    return self.ffn(normed_residual) + residual_after_attention, new_kv_cache

if __name__ == "__main__":
  config = TinformerConfig()
  block = DecoderBlock(
      config.D_model, config.D_k, config.D_v, config.H
  )
  X = jax.random.normal(
      jax.random.PRNGKey(0),
      (config.B, config.T, config.D_model)
  )
  mask = jnp.tril(
      jnp.ones((config.T, config.T)), dtype=bool
  )
  print("Decoder output shape: ", block(X, mask)[0].shape)
