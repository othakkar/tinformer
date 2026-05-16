import time

import jax
import jax.numpy as jnp

from src.tinformer import Tinformer, TinformerConfig

def cached_generate(model, input_prompt_token_ids, num_tokens_to_generate):
  # Prefill stage: build the KV cache
  logits, kv_caches = model.forward(input_prompt_token_ids) # (B, T, vocab_size)
  probs = jax.nn.softmax(logits[:, -1, :], axis=-1) # (B, vocab_size)
  next_token_ids = jnp.argmax(probs, axis=-1, keepdims=True) # (B, 1)

  generated_tokens = [next_token_ids]
  for _ in range(num_tokens_to_generate - 1):
    logits, kv_caches = model.forward(next_token_ids, kv_caches=kv_caches)
    probs = jax.nn.softmax(logits[:, -1, :], axis=-1) # (B, vocab_size)
    next_token_ids = jnp.argmax(probs, axis=-1, keepdims=True) # (B, 1)
    generated_tokens.append(next_token_ids)
  return jnp.concatenate(generated_tokens, axis=1) # (B, num_tokens_to_generate)


def naive_generate(model, input_prompt_token_ids, num_tokens_to_generate):
  for _ in range(num_tokens_to_generate):
    logits, _ = model.forward(input_prompt_token_ids)  # (B, T, vocab_size)
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
  
  # Naive generate
  start_time = time.time()
  generated_token_ids = naive_generate(model, input_prompt_token_ids, num_tokens_to_generate)
  print(f"Generated token IDs shape: {generated_token_ids.shape}, Time taken to generate: {time.time() - start_time} secs")

  # Cached generate
  start_time = time.time()
  generated_token_ids = cached_generate(model, input_prompt_token_ids, num_tokens_to_generate)
  print(f"Generated token IDs shape: {generated_token_ids.shape}, Time taken to generate: {time.time() - start_time} secs")

