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

  def forward(self, token_ids):
    T = token_ids.shape[-1]
    causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    X = self.tok_embeddings[token_ids] + self.pos_embeddings[:T, :]
    for block in self.layers:
      X = block(X, attention_mask=causal_mask)

    X_norm = self.final_ln(X)
    return jnp.matmul(X_norm, self.W_logits)


if __name__ == "__main__":
  config = TinformerConfig()
  model = Tinformer(config)

  # Dummy input token IDs
  B = config.B
  T = config.T
  token_ids = jax.random.randint(jax.random.PRNGKey(0), (B, T), minval=0, maxval=config.vocab_size)

  logits = model.forward(token_ids)
  print("Logits shape: ", logits.shape)
