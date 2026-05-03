# Tinformer
A tiny decoder-only transformer in JAX where every component, from scaled dot-product attention to autoregressive generation, is implemented from scratch. Built to trace every tensor shape through every layer, not to produce coherent text.

## Repo structure
 
```
.gitignore
LICENSE
README.md
generate.py                     # Generate function for autoregressive token generation
requirements.txt
src/
  config.py                     # TinformerConfig dataclass
  attention.py                  # Scaled dot-product attention + multi-head attention
  layernorm.py                  # Layer normalization
  ffn.py                        # Feed-forward network
  decoder.py                    # Decoder block (LN → MHA → residual → LN → FFN → residual)
  tinformer.py                  # Full model + generate function
tests/
  conftest.py                   # Pytest configuration
  test_shapes.py                # Tensor shape verification
  test_causal_masking.py        # Causal mask validation
  test_attention_stability.py   # Numerical stability tests
```
 
## Quickstart
 
```bash
git clone https://github.com/othakkar/tinformer.git
cd tinformer
pip install -r requirements.txt
python -m generate
```
 
## Run tests
 
```bash
pytest tests/
```

## License
 
Apache 2.0
