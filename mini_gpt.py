from decoder import decoder_block, Attention, FFN
from layer_norm import layer_norm
from config import GPTConfig

import jax
import jax.numpy as jnp

class MiniGPT:
  def __init__(self, config: GPTConfig):
    self.config = config
    self.params = self._init_params()
  
  def _init_params(self):
    c = self.config
    # Token embeddings
    tok_embeddings = jax.random.normal(jax.random.PRNGKey(0), (c.vocab_size, c.D_model))
    # Positional embeddings
    pos_embeddings = jax.random.normal(jax.random.PRNGKey(1), (c.max_seq_len, c.D_model))

    layers = []
    for i in range(c.N):
      # Random attention weight matrices
      W_q = jax.random.normal(jax.random.PRNGKey(i * 10 + 1), (c.D_model, c.H * c.D_k))
      W_k = jax.random.normal(jax.random.PRNGKey(i * 10 + 2), (c.D_model, c.H * c.D_k))
      W_v = jax.random.normal(jax.random.PRNGKey(i * 10 + 3), (c.D_model, c.H * c.D_v))
      W_o = jax.random.normal(jax.random.PRNGKey(i * 10 + 4), (c.H * c.D_v, c.D_model))

      # Random ffn weight matrices
      W1 = jax.random.normal(jax.random.PRNGKey(i * 10 + 5), (c.D_model, 4 * c.D_model))
      b1 = jax.random.normal(jax.random.PRNGKey(i * 10 + 6), (4 * c.D_model,))
      W2 = jax.random.normal(jax.random.PRNGKey(i * 10 + 7), (4 * c.D_model, c.D_model))
      b2 = jax.random.normal(jax.random.PRNGKey(i * 10 + 8), (c.D_model,))

      # Random LayerNorm parameter initialization
      ln1_scale = jnp.ones((c.D_model,))
      ln1_bias = jnp.zeros((c.D_model,))
      ln2_scale = jnp.ones((c.D_model,))
      ln2_bias = jnp.zeros((c.D_model,))

      attn_params = Attention(W_q, W_k, W_v, W_o, c.H, ln1_scale, ln1_bias)
      ffn_params = FFN(W1, b1, W2, b2, ln2_scale, ln2_bias)

      layers.append((attn_params, ffn_params))
    
    # Final LN scale and bias
    ln_scale = jnp.ones((c.D_model, ))
    ln_bias = jnp.zeros((c.D_model, ))
    # W_logits
    W_logits = jax.random.normal(jax.random.PRNGKey(c.N * 10 + 1), (c.D_model, c.vocab_size))

    return (tok_embeddings, pos_embeddings, layers, ln_scale, ln_bias, W_logits)

  def forward(self, token_ids):
    tok_embeddings, pos_embeddings, layers, ln_scale, ln_bias, W_logits = self.params  
    T = token_ids.shape[1]
    causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))  # (T, T), True = keep

    X = tok_embeddings[token_ids] + pos_embeddings[:T, :]
    for attn_params, ffn_params in layers:
      X = decoder_block(X, attn_params=attn_params, ffn_params=ffn_params, attention_mask=causal_mask)

    X_norm = layer_norm(X, ln_scale, ln_bias)
    return jnp.matmul(X_norm, W_logits)


if __name__ == "__main__":
  config = GPTConfig()
  model = MiniGPT(config)

  # Dummy input token IDs
  B = config.B
  T = config.T
  token_ids = jax.random.randint(jax.random.PRNGKey(0), (B, T), minval=0, maxval=config.vocab_size)

  logits = model.forward(token_ids)
  print("Logits shape: ", logits.shape)