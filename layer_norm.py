import jax
import jax.numpy as jnp

def layer_norm(X, scale, bias, eps=1e-6):
  # X: (B, T, D_model)
  # scale: (D_model,)
  # bias: (D_model,)

  mean_X = jnp.mean(X, axis=-1, keepdims=True) # (B, T, 1)
  var_X = jnp.var(X, axis=-1, keepdims=True) # (B, T, 1)

  X_norm = (X - mean_X) / jnp.sqrt(var_X + eps) # (B, T, D_model)
  output = scale * X_norm + bias # (B, T, D_model)
  return output 

B = 16  # batch size
T = 128  # sequence length
D_model = 512  # embedding dimension

# Random input data and weight matrices
X = jax.random.normal(jax.random.PRNGKey(0), (B, T, D_model))
scale = jnp.ones((D_model,))
bias = jnp.zeros((D_model,))

output = layer_norm(X, scale, bias)
print(output.shape) # (B, T, D_model)