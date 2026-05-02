# Tinformer
A tiny decoder-only transformer built from scratch in JAX. No framework magic, no pretrained weights. Every component (attention, layer norm, FFN, KV caching) is implemented by hand to understand the mechanics of autoregressive generation.

## Repo structure
 
```
src/
  config.py                     # TinformerConfig dataclass
  attention.py                  # Scaled dot-product attention + multi-head attention
  layernorm.py                  # Layer normalization
  ffn.py                        # Feed-forward network
  decoder.py                    # Decoder block (LN → MHA → residual → LN → FFN → residual)
  tinformer.py                  # Full model + generate function
tests/
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
