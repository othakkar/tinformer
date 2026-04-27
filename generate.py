from mini_gpt import MiniGPT
from config import GPTConfig

import jax
import jax.numpy as jnp
import time

def naive_generate(model, input_prompt_token_ids, num_tokens_to_generate):
  for i in range(num_tokens_to_generate):
    logits = model.forward(input_prompt_token_ids)  # (B, T, vocab_size)
    probs = jax.nn.softmax(logits[:, -1, :])  # (B, vocab_size)
    next_token_ids = jnp.argmax(probs, axis=-1)  # (B,)

    input_prompt_token_ids = jnp.concatenate(
        [input_prompt_token_ids, next_token_ids[:, None]], axis=1
    )  # (B, T+1)
  return input_prompt_token_ids

if __name__ == "__main__":
  config = GPTConfig()
  model = MiniGPT(config)
  num_tokens_to_generate = 100

  # Dummy input token IDs
  input_prompt_token_ids = jax.random.randint(jax.random.PRNGKey(0), (config.B, config.T), minval=0, maxval=config.vocab_size)

  start_time = time.time()
  generated_token_ids = naive_generate(model, input_prompt_token_ids, num_tokens_to_generate)
  end_time = time.time()
  print(f"Generated token IDs shape: {generated_token_ids.shape}")
  print(f"Time taken to generate {num_tokens_to_generate} tokens: {end_time - start_time:.2f} seconds")