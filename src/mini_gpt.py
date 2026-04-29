from decoder import DecoderBlock
from layernorm import LayerNorm
from config import GPTConfig

import jax
import jax.numpy as jnp

class MiniGPT:
  def __init__(self, config: GPTConfig):
    self.config = config
    keys = jax.random.split(jax.random.PRNGKey(0), self.config.N + 3)

    self.tok_embeddings = jax.random.normal(keys[0], (config.vocab_size, config.D_model))
    self.pos_embeddings = jax.random.normal(keys[1], (config.max_seq_len, config.D_model))

    self.layers = [
      DecoderBlock(config.D_model, config.D_k, config.D_v, config.H, key=keys[i + 2]) for i in range(config.N)
    ]

    self.final_ln = LayerNorm(config.D_model)
    self.W_logits = jax.random.normal(keys[-1], (config.D_model, config.vocab_size))    

  def forward(self, token_ids, kv_caches=None):
    T = token_ids.shape[-1]
    start_pos = 0 if kv_caches is None else kv_caches[0][0].shape[2] # cached sequence length
    S = start_pos + T
    causal_mask = jnp.tril(jnp.ones((S, S), dtype=bool))[-T:, :] # (T, S)
    X = self.tok_embeddings[token_ids] + self.pos_embeddings[start_pos:start_pos + T, :]
    all_new_caches = []
    for i, block in enumerate(self.layers):
      cache_i = kv_caches[i] if kv_caches is not None else None
      X, new_kv_cache = block(X, attention_mask=causal_mask, kv_cache=cache_i)
      all_new_caches.append(new_kv_cache)

    X_norm = self.final_ln(X)
    return jnp.matmul(X_norm, self.W_logits), all_new_caches


if __name__ == "__main__":
  config = GPTConfig()
  model = MiniGPT(config)

  # Dummy input token IDs
  B = config.B
  T = config.T
  token_ids = jax.random.randint(jax.random.PRNGKey(0), (B, T), minval=0, maxval=config.vocab_size)

  logits, caches = model.forward(token_ids)
  print("Logits shape: ", logits.shape)