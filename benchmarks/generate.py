import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))  # Add the src directory to the Python path

from tinformer import Tinformer
from config import GPTConfig

import jax
import jax.numpy as jnp
import time

def naive_generate(model, input_prompt_token_ids, num_tokens_to_generate):
  for i in range(num_tokens_to_generate):
    logits, _ = model.forward(input_prompt_token_ids)  # (B, T, vocab_size)
    probs = jax.nn.softmax(logits[:, -1, :])  # (B, vocab_size)
    next_token_ids = jnp.argmax(probs, axis=-1)  # (B, )

    input_prompt_token_ids = jnp.concatenate(
        [input_prompt_token_ids, next_token_ids[:, None]], axis=1
    )  # (B, T+1)
  return input_prompt_token_ids[:, -num_tokens_to_generate:]  # Return only the generated tokens

def cached_generate(model, input_prompt_token_ids, num_tokens_to_generate):
  # Prefill stage. Build the KV Cache
  logits, kv_caches = model.forward(input_prompt_token_ids, kv_caches=None) # (B, T, vocab_size)
  probs = jax.nn.softmax(logits[:, -1, :]) # (B, vocab_size)
  next_token_ids = jnp.argmax(probs, axis=-1, keepdims=True) # (B, 1)

  # Decode stage. One token at a time.
  generated_tokens = [next_token_ids]
  for i in range(num_tokens_to_generate - 1):
    logits, kv_caches = model.forward(next_token_ids, kv_caches=kv_caches) # (B, 1, vocab_size)
    probs = jax.nn.softmax(logits[:, -1, :]) # (B, vocab_size)
    next_token_ids = jnp.argmax(probs, axis=-1, keepdims=True) # (B, 1)
    generated_tokens.append(next_token_ids)
  
  return jnp.concatenate(generated_tokens, axis=1) # (B, num_tokens_to_generate)

if __name__ == "__main__":
  config = GPTConfig()
  model = Tinformer(config)
  num_tokens_to_generate = 100

  # Dummy input token IDs
  input_prompt_token_ids = jax.random.randint(jax.random.PRNGKey(0), (config.B, config.T), minval=0, maxval=config.vocab_size)

  start_time = time.time()
  naively_generated_token_ids = naive_generate(model, input_prompt_token_ids, num_tokens_to_generate)
  end_time = time.time()
  print(f"Generated token IDs shape: {naively_generated_token_ids.shape}")
  print(f"naive_generate: Time taken to generate {num_tokens_to_generate} tokens: {end_time - start_time:.2f} seconds")

  start_time = time.time()
  cached_generated_token_ids = cached_generate(model, input_prompt_token_ids, num_tokens_to_generate)
  end_time = time.time()
  print(f"Generated token IDs shape: {cached_generated_token_ids.shape}")
  print(f"cached_generate: Time taken to generate {num_tokens_to_generate} tokens: {end_time - start_time:.2f} seconds")

  # Verify correctness with KV cache
  print("Generated tokens match between naive and cached generation: ", jnp.allclose(naively_generated_token_ids, cached_generated_token_ids))
