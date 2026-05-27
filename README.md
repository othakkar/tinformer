# Tinformer
A tiny decoder-only transformer in JAX where every component, from scaled dot-product attention to autoregressive generation, is implemented from scratch. Supports Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) via a configurable `num_kv_heads` parameter. Built to trace every tensor shape through every layer, not to produce coherent text.

## Tags

| Tag | Description | Article |
|-----|-------------|---------|
| [`tinformer-from-scratch`](https://github.com/othakkar/tinformer/tree/tinformer-from-scratch) | Base transformer implementation | [Tinformer: Building a Tiny Transformer from Scratch in JAX](https://open.substack.com/pub/computebound/p/tinformer-building-a-tiny-transformer?r=87stqj&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) |
| [`kv-caching`](https://github.com/othakkar/tinformer/tree/kv-caching) | KV caching with benchmarks | [KV Caching from Scratch in JAX](https://open.substack.com/pub/computebound/p/kv-caching-from-scratch-in-jax-eliminating?r=87stqj&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) |
| [`mqa-gqa`](https://github.com/othakkar/tinformer/tree/mqa-gqa) | Multi-Query & Grouped-Query Attention | [MQA and GQA: Shrinking the KV Cache in Tinformer](https://open.substack.com/pub/computebound/p/mqa-and-gqa-shrinking-the-kv-cache?r=87stqj&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) |

## Repo structure
 
```
.gitignore
LICENSE
README.md
generate.py                          # Naive + cached autoregressive generation
requirements.txt
src/
  config.py                          # TinformerConfig dataclass (incl. num_kv_heads for MQA/GQA)
  attention.py                       # Scaled dot-product attention + MHA/MQA/GQA (with KV cache)
  layernorm.py                       # Layer normalization
  ffn.py                             # Feed-forward network
  decoder.py                         # Decoder block (LN → MHA → residual → LN → FFN → residual)
  tinformer.py                       # Full model with KV cache support
benchmarks/
  benchmark_kv_caching.py            # Naive vs cached generation benchmarks
tests/
  test_shapes.py                     # Tensor shape verification
  test_causal_masking.py             # Causal mask validation
  test_attention_stability.py        # Numerical stability tests
  test_cached_generate.py            # KV cache correctness tests
  test_mqa_gqa.py                    # MQA/GQA configuration tests
```
 
## Quickstart
 
```bash
git clone https://github.com/othakkar/tinformer.git
cd tinformer
pip install -r requirements.txt
python -m generate
```

## Run benchmarks
```bash
python -m benchmarks.benchmark_kv_caching
```
 
## Run tests
 
```bash
pytest tests/
```

## License
 
Apache 2.0
