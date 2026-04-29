import jax
import jax.numpy as jnp

from attention import MultiHeadAttention
from layernorm import LayerNorm
from ffn import FFN

class DecoderBlock:
  def __init__(self, D_model, D_k, D_v, H, key=jax.random.PRNGKey(0)):
    keys = jax.random.split(key, 3)
    self.attn = MultiHeadAttention(D_model, D_k, D_v, H, key=keys[0])
    self.ln1 = LayerNorm(D_model)
    self.ln2 = LayerNorm(D_model)

    self.ffn = FFN(D_model, key=keys[1])
  
  def __call__(self, hidden_states, attention_mask=None, kv_cache=None):
    normed_hidden_states = self.ln1(hidden_states)
    attention_out, new_kv_cache = self.attn(normed_hidden_states, mask=attention_mask, kv_cache=kv_cache)
    residual_after_attention = attention_out + hidden_states
    normed_residual = self.ln2(residual_after_attention)
    return self.ffn(normed_residual) + residual_after_attention, new_kv_cache

if __name__ == "__main__":
  from config import GPTConfig
  config = GPTConfig()
  block = DecoderBlock(config.D_model, config.D_k, config.D_v, config.H)
  X = jax.random.normal(jax.random.PRNGKey(0), (config.B, config.T, config.D_model))
  mask = jnp.tril(jnp.ones((config.T, config.T)), dtype=bool)
  out, kv_cache = block(X, mask)
  print("Decoder output shape: ", out.shape)