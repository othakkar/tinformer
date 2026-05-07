from .decoder import DecoderBlock
from .layernorm import LayerNorm
from .config import TinformerConfig

import jax
import jax.numpy as jnp

class Tinformer:
  def __init__(self, config: TinformerConfig, key=jax.random.PRNGKey(0)):
    keys = jax.random.split(key, config.N + 3)

    self.tok_embeddings = jax.random.normal(keys[0], (config.vocab_size, config.D_model))
    self.pos_embeddings = jax.random.normal(keys[1], (config.max_seq_len, config.D_model))

    self.layers = [
      DecoderBlock(
          config.D_model, config.D_k, config.D_v, config.H, key=keys[i + 2]
      ) for i in range(config.N)
    ]

    self.final_ln = LayerNorm(config.D_model)
    self.W_logits = jax.random.normal(keys[-1], (config.D_model, config.vocab_size))    

  def forward(self, token_ids, kv_caches=None):
    T = token_ids.shape[-1]   # T=1 when kv_caches is not None
    start_pos = 0 if kv_caches is None else kv_caches[0][0].shape[2] # kv_caches[0][0].shape[2] represents the seq len so far
    S = start_pos + T
    # Causal mask: lower triangular (prefill) or row (decode)
    causal_mask = jnp.tril(jnp.ones((S, S), dtype=bool))[-T:, :]
    # Add position embeddings (from 0:T for prefill, start_pos:start_pos+T for decode)
    X = self.tok_embeddings[token_ids] + self.pos_embeddings[start_pos:start_pos + T, :]

    all_new_caches = []
    for i, block in enumerate(self.layers):
      cache_i = kv_caches[i] if kv_caches is not None else None
      X, new_cache = block(X, attention_mask=causal_mask, kv_cache=cache_i)
      all_new_caches.append(new_cache)

    X_norm = self.final_ln(X)
    return jnp.matmul(X_norm, self.W_logits), all_new_caches


if __name__ == "__main__":
  config = TinformerConfig()
  model = Tinformer(config)

  # Dummy input token IDs
  B = config.B
  T = config.T
  token_ids = jax.random.randint(jax.random.PRNGKey(0), (B, T), minval=0, maxval=config.vocab_size)

  logits, _ = model.forward(token_ids)
  print("Logits shape: ", logits.shape)
