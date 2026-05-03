import jax
import jax.numpy as jnp

from src.tinformer import Tinformer, TinformerConfig

def generate(model, input_prompt_token_ids, num_tokens_to_generate):
  for _ in range(num_tokens_to_generate):
    logits = model.forward(input_prompt_token_ids)  # (B, T, vocab_size)
    probs = jax.nn.softmax(logits[:, -1, :])  # (B, vocab_size)
    next_token_ids = jnp.argmax(probs, axis=-1)  # (B, )

    input_prompt_token_ids = jnp.concatenate(
        [input_prompt_token_ids, next_token_ids[:, None]], axis=1
    )  # (B, T+1)
  return input_prompt_token_ids[:, -num_tokens_to_generate:]  # Return only the generated tokens

if __name__ == "__main__":
  config = TinformerConfig()
  model = Tinformer(config)
  num_tokens_to_generate = 5

  # Dummy input token IDs
  input_prompt_token_ids = jax.random.randint(jax.random.PRNGKey(0), (config.B, config.T), minval=0, maxval=config.vocab_size)
  generated_token_ids = generate(model, input_prompt_token_ids, num_tokens_to_generate)
  print(f"Generated token IDs shape: {generated_token_ids.shape}")