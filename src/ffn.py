import jax
import jax.numpy as jnp

from .config import TinformerConfig

class FFN:
  def __init__(self, D_model, key=jax.random.PRNGKey(0)):
    keys = jax.random.split(key, 4)
    self.W1 = jax.random.normal(keys[0], (D_model, 4 * D_model))
    self.b1 = jax.random.normal(keys[1], (4 * D_model,))
    self.W2 = jax.random.normal(keys[2], (4 * D_model, D_model))
    self.b2 = jax.random.normal(keys[3], (D_model,))

  def __call__(self, X):
    hidden = jax.nn.gelu(jnp.dot(X, self.W1) + self.b1)
    return jnp.dot(hidden, self.W2) + self.b2

if __name__ == "__main__":
  config = TinformerConfig()
  ffn = FFN(config.D_model)
  X = jax.random.normal(jax.random.PRNGKey(0), (config.B, config.T, config.D_model))
  print("FFN output shape: ", ffn(X).shape)
