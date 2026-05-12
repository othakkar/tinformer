import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.tinformer import Tinformer
from src.config import TinformerConfig

import jax
import jax.numpy as jnp
import time

def naive_generate(model, input_prompt_token_ids, num_tokens_to_generate):
  for i in range(num_tokens_to_generate):
    logits, _ = model.forward(input_prompt_token_ids)
    probs = jax.nn.softmax(logits[:, -1, :])
    next_token_ids = jnp.argmax(probs, axis=-1)
    input_prompt_token_ids = jnp.concatenate(
        [input_prompt_token_ids, next_token_ids[:, None]], axis=1
    )
  return input_prompt_token_ids[:, -num_tokens_to_generate:]

def cached_generate(model, input_prompt_token_ids, num_tokens_to_generate):
  logits, kv_caches = model.forward(input_prompt_token_ids, kv_caches=None)
  probs = jax.nn.softmax(logits[:, -1, :])
  next_token_ids = jnp.argmax(probs, axis=-1, keepdims=True)

  generated_tokens = [next_token_ids]
  for i in range(num_tokens_to_generate - 1):
    logits, kv_caches = model.forward(next_token_ids, kv_caches=kv_caches)
    probs = jax.nn.softmax(logits[:, -1, :])
    next_token_ids = jnp.argmax(probs, axis=-1, keepdims=True)
    generated_tokens.append(next_token_ids)

  return jnp.concatenate(generated_tokens, axis=1)

def run_benchmark(config, num_tokens_to_generate, warmup=True):
  """Run a single benchmark for a given config and generation length."""
  model = Tinformer(config)
  prompt = jax.random.randint(jax.random.PRNGKey(0), (config.B, config.T), 0, config.vocab_size)

  # Warmup run (JIT compilation)
  if warmup:
    _ = cached_generate(model, prompt, 1)

  # Naive generation
  start = time.time()
  naive_out = naive_generate(model, prompt, num_tokens_to_generate)
  naive_time = time.time() - start

  # Cached generation
  start = time.time()
  cached_out = cached_generate(model, prompt, num_tokens_to_generate)
  cached_time = time.time() - start

  # Correctness check
  correct = jnp.allclose(naive_out.astype(jnp.float32), 
                       cached_out.astype(jnp.float32), atol=1e-5)

  return {
    "naive_time": naive_time,
    "cached_time": cached_time,
    "speedup": naive_time / cached_time if cached_time > 0 else float('inf'),
    "correct": correct,
  }

def print_results_table(header, rows):
  """Pretty-print a results table."""
  col_widths = [max(len(str(row[i])) for row in [header] + rows) + 2 for i in range(len(header))]
  fmt = "".join(f"{{:<{w}}}" for w in col_widths)
  print(fmt.format(*header))
  print("-" * sum(col_widths))
  for row in rows:
    print(fmt.format(*row))
  print()

def sweep_generation_length():
  """Sweep: vary number of tokens to generate."""
  print("=" * 70)
  print("SWEEP 1: Varying num_tokens_to_generate")
  print("Config: D_model=512, N=10, H=8, T=128 (prompt), B=1")
  print("=" * 70)

  config = TinformerConfig(B=1, T=128, D_model=512, N=10, H=8)
  gen_lengths = [10, 25, 50, 100, 200]

  header = ["Tokens Generated", "Naive (s)", "Cached (s)", "Speedup", "Correct"]
  rows = []
  for n_tokens in gen_lengths:
    result = run_benchmark(config, n_tokens)
    rows.append([
      str(n_tokens),
      f"{result['naive_time']:.3f}",
      f"{result['cached_time']:.3f}",
      f"{result['speedup']:.2f}x",
      str(result['correct']),
    ])
  print_results_table(header, rows)

def sweep_prompt_length():
  """Sweep: vary prompt length T."""
  print("=" * 70)
  print("SWEEP 2: Varying prompt length (T)")
  print("Config: D_model=512, N=10, H=8, gen=50, B=1")
  print("=" * 70)

  prompt_lengths = [32, 64, 128, 256, 512]
  num_tokens_to_generate = 50

  header = ["Prompt Length (T)", "Naive (s)", "Cached (s)", "Speedup", "Correct"]
  rows = []
  for T in prompt_lengths:
    config = TinformerConfig(B=1, T=T, D_model=512, N=10, H=8)
    result = run_benchmark(config, num_tokens_to_generate)
    rows.append([
      str(T),
      f"{result['naive_time']:.3f}",
      f"{result['cached_time']:.3f}",
      f"{result['speedup']:.2f}x",
      str(result['correct']),
    ])
  print_results_table(header, rows)

def sweep_num_layers():
  """Sweep: vary number of decoder layers N."""
  print("=" * 70)
  print("SWEEP 3: Varying number of layers (N)")
  print("Config: D_model=512, H=8, T=128, gen=50, B=1")
  print("=" * 70)

  layer_counts = [2, 4, 6, 10]
  num_tokens_to_generate = 50

  header = ["Num Layers (N)", "Naive (s)", "Cached (s)", "Speedup", "Correct"]
  rows = []
  for N in layer_counts:
    config = TinformerConfig(B=1, T=128, D_model=512, N=N, H=8)
    result = run_benchmark(config, num_tokens_to_generate)
    rows.append([
      str(N),
      f"{result['naive_time']:.3f}",
      f"{result['cached_time']:.3f}",
      f"{result['speedup']:.2f}x",
      str(result['correct']),
    ])
  print_results_table(header, rows)

def sweep_model_dimension():
  """Sweep: vary model dimension D_model."""
  print("=" * 70)
  print("SWEEP 4: Varying model dimension (D_model)")
  print("Config: N=6, H=8, T=128, gen=50, B=1")
  print("=" * 70)

  dims = [128, 256, 512]
  num_tokens_to_generate = 50

  header = ["D_model", "D_k", "Naive (s)", "Cached (s)", "Speedup", "Correct"]
  rows = []
  for D in dims:
    D_k = D // 8  # Keep D_k proportional
    config = TinformerConfig(B=1, T=128, D_model=D, D_k=D_k, D_v=D_k, N=6, H=8)
    result = run_benchmark(config, num_tokens_to_generate)
    rows.append([
      str(D),
      str(D_k),
      f"{result['naive_time']:.3f}",
      f"{result['cached_time']:.3f}",
      f"{result['speedup']:.2f}x",
      str(result['correct']),
    ])
  print_results_table(header, rows)

def sweep_per_step_latency():
  """Measure per-step latency for naive vs. cached to show constant vs. growing cost."""
  print("=" * 70)
  print("SWEEP 5: Per-step latency (naive grows, cached stays flat)")
  print("Config: D_model=512, N=10, H=8, T=64, B=1")
  print("=" * 70)

  config = TinformerConfig(B=1, T=64, D_model=512, N=10, H=8)
  model = Tinformer(config)
  prompt = jax.random.randint(jax.random.PRNGKey(0), (config.B, config.T), 0, config.vocab_size)
  num_steps = 50

  # Naive: measure each step
  naive_step_times = []
  current_ids = prompt
  for i in range(num_steps):
    start = time.time()
    logits, _ = model.forward(current_ids)
    probs = jax.nn.softmax(logits[:, -1, :])
    next_token = jnp.argmax(probs, axis=-1)
    current_ids = jnp.concatenate([current_ids, next_token[:, None]], axis=1)
    naive_step_times.append(time.time() - start)

  # Cached: measure each step
  cached_step_times = []
  start = time.time()
  logits, kv_caches = model.forward(prompt, kv_caches=None)
  probs = jax.nn.softmax(logits[:, -1, :])
  next_token = jnp.argmax(probs, axis=-1, keepdims=True)
  cached_step_times.append(time.time() - start)  # prefill step

  for i in range(num_steps - 1):
    start = time.time()
    logits, kv_caches = model.forward(next_token, kv_caches=kv_caches)
    probs = jax.nn.softmax(logits[:, -1, :])
    next_token = jnp.argmax(probs, axis=-1, keepdims=True)
    cached_step_times.append(time.time() - start)

  header = ["Step", "Naive (ms)", "Cached (ms)"]
  rows = []
  for i in range(0, num_steps, 5):  # Print every 5th step
    rows.append([
      str(i + 1),
      f"{naive_step_times[i] * 1000:.1f}",
      f"{cached_step_times[i] * 1000:.1f}",
    ])
  print_results_table(header, rows)

def kv_cache_memory_table():
  """Show KV cache memory footprint as sequence length grows."""
  print("=" * 70)
  print("SWEEP 6: KV Cache Memory Footprint")
  print("Config: D_model=512, D_k=32, D_v=32, H=8, N=10, B=1, float32")
  print("=" * 70)

  N = 10
  H = 8
  D_k = 32
  D_v = 32
  B = 1
  bytes_per_float = 4  # float32

  header = ["Seq Length", "Cache Size (MB)", "Per Layer (MB)"]
  rows = []
  for T_cached in [64, 128, 256, 512, 1024]:
    per_layer = 2 * B * H * T_cached * D_k * bytes_per_float  # 2 for K and V
    total = N * per_layer
    rows.append([
      str(T_cached),
      f"{total / (1024**2):.2f}",
      f"{per_layer / (1024**2):.3f}",
    ])
  print_results_table(header, rows)


if __name__ == "__main__":
  print("\nKV Cache Benchmark Suite\n")

  kv_cache_memory_table()
  sweep_generation_length()
  sweep_prompt_length()
  sweep_num_layers()
  sweep_model_dimension()
  sweep_per_step_latency()

  print("All benchmarks complete.")
