import jax
import jax.numpy as jnp

from attention import MultiHeadAttention
from layer_norm import LayerNorm

class DecoderBlock:
  def __init__(self, D_model, D_k, D_v, H, key=jax.random.PRNGKey(0)):
    keys = jax.random.split(key, 3)
    self.attn = MultiHeadAttention(D_model, D_k, D_v, H, key=keys[0])
    self.ln1 = LayerNorm(D_model)
    self.ln2 = LayerNorm(D_model)

    self.W1 = jax.random.normal(keys[1], (D_model, 4 * D_model))
    self.b1 = jax.random.normal(keys[1], (4 * D_model,))
    self.W2 = jax.random.normal(keys[2], (4 * D_model, D_model))
    self.b2 = jax.random.normal(keys[2], (D_model,))
  
  def __call__(self, hidden_states, attention_mask=None):
    normed_hidden_states = self.ln1(hidden_states)
    attention_out = self.attn(normed_hidden_states, mask=attention_mask)
    residual_after_attention = attention_out + hidden_states
    normed_residual = self.ln2(residual_after_attention)
    ffn_hidden = jnp.dot(normed_residual, self.W1) + self.b1
    ffn_act = jax.nn.gelu(ffn_hidden)
    ffn_out = jnp.dot(ffn_act, self.W2) + self.b2
    return ffn_out + residual_after_attention

if __name__ == "__main__":
  from config import GPTConfig
  config = GPTConfig()
  block = DecoderBlock(config.D_model, config.D_k, config.D_v, config.H)
  X = jax.random.normal(jax.random.PRNGKey(0), (config.B, config.T, config.D_model))
  mask = jnp.tril(jnp.ones((config.T, config.T)), dtype=bool)
  print("Decoder output shape: ", block(X, mask).shape)