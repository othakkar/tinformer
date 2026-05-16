import jax
import jax.numpy as jnp

from src.config import TinformerConfig

class LayerNorm:
  def __init__(self, D_model, eps=1e-6):
    self.scale = jnp.ones((D_model,))
    self.bias = jnp.zeros((D_model,))
    self.eps = eps

  def __call__(self, X):
    mean_X = jnp.mean(X, axis=-1, keepdims=True)  # (B, T, 1)
    var_X = jnp.var(X, axis=-1, keepdims=True)  # (B, T, 1)
    X_norm = (X - mean_X) / jnp.sqrt(
        var_X + self.eps
    )  # (B, T, D_model)
    output = self.scale * X_norm + self.bias  # (B, T, D_model)
    return output

if __name__ == "__main__":
  config = TinformerConfig()
  ln = LayerNorm(config.D_model)
  X = jax.random.normal(jax.random.PRNGKey(0), (config.B, config.T, config.D_model))
  print("LayerNorm output shape: ", ln(X).shape)
