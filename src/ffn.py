import jax
import jax.numpy as jnp

class FFN:
  def __init__(self, D_model, key=jax.random.PRNGKey(0)):
    keys = jax.random.split(key, 2)
    self.W1 = jax.random.normal(keys[0], (D_model, 4 * D_model))
    self.b1 = jax.random.normal(keys[0], (4 * D_model,))
    self.W2 = jax.random.normal(keys[1], (4 * D_model, D_model))
    self.b2 = jax.random.normal(keys[1], (D_model,))

  def __call__(self, X):
    return jnp.dot(jax.nn.gelu(jnp.dot(X, self.W1) + self.b1), self.W2) + self.b2

if __name__ == "__main__":
  from .config import GPTConfig
  config = GPTConfig()
  ffn = FFN(config.D_model)
  X = jax.random.normal(jax.random.PRNGKey(0), (config.B, config.T, config.D_model))
  print("FFN output shape: ", ffn(X).shape)
