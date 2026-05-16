import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.tinformer import Tinformer
from src.config import TinformerConfig
from generate import naive_generate, cached_generate

import jax
import jax.numpy as jnp
import time

def run_benchmark(config, num_tokens_to_generate, warmup=True):
  """Run a single benchmark for a given config and generation length."""
  model = Tinformer(config)
  prompt = jax.random.randint(jax.random.PRNGKey(0), (config.B, config.T), 0, config.vocab_size)

  # Warmup run: use full generation length to compile all intermediate shapes
  if warmup:
    naive_generate(model, prompt, num_tokens_to_generate).block_until_ready()
    cached_generate(model, prompt, num_tokens_to_generate).block_until_ready()

  # Naive generation
  start = time.time()
  naive_out = naive_generate(model, prompt, num_tokens_to_generate)
  naive_out.block_until_ready()
  naive_time = time.time() - start

  # Cached generation
  start = time.time()
  cached_out = cached_generate(model, prompt, num_tokens_to_generate)
  cached_out.block_until_ready()
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

def sweep_prompt_length():
  """Sweep: vary prompt length T."""
  print("=" * 70)
  print("Varying prompt length (T)")
  print("Config: D_model=512, N=10, H=8, gen=100, B=1")
  print("=" * 70)

  prompt_lengths = [32, 64, 128, 256, 512]
  num_tokens_to_generate = 100

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

def sweep_per_step_latency():
  """Measure per-step latency for naive vs. cached to show constant vs. growing cost."""
  print("=" * 70)
  print("Per-step latency (naive grows, cached stays flat)")
  print("Config: D_model=512, N=10, H=8, T=64, B=1")
  print("=" * 70)

  config = TinformerConfig(B=1, T=64, D_model=512, N=10, H=8)
  gen_lengths = [i*10 for i in range(1, 11)]  # 10, 20, ..., 100 tokens

  header = ["Tokens Generated", "Naive per-step (ms)", "Cached per-step (ms)"]
  rows = []
  for n_tokens in gen_lengths:
    result = run_benchmark(config, n_tokens)
    rows.append([
      str(n_tokens),
      f"{result['naive_time'] / n_tokens * 1000:.1f}",
      f"{result['cached_time'] / n_tokens * 1000:.1f}",
    ])
  print_results_table(header, rows)

def kv_cache_memory_table():
  """Show KV cache memory footprint as sequence length grows."""
  print("=" * 70)
  print("KV Cache Memory Footprint")
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

  # Global warmup to absorb JAX startup cost
  dummy_config = TinformerConfig(B=1, T=32, D_model=128, N=2, H=4)
  dummy_model = Tinformer(dummy_config)
  dummy_prompt = jax.random.randint(jax.random.PRNGKey(0), (1, 32), 0, 100)
  naive_generate(dummy_model, dummy_prompt, 1).block_until_ready()
  cached_generate(dummy_model, dummy_prompt, 1).block_until_ready()
  print("Warmup complete.\n")

  sweep_prompt_length()
  sweep_per_step_latency()
  kv_cache_memory_table()

  print("All benchmarks complete.")
