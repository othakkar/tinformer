from dataclasses import dataclass

@dataclass
class TinformerConfig:
  B: int = 16
  T: int = 128
  D_model: int = 512
  D_k: int = 32
  D_v: int = 32
  H: int = 8
  vocab_size: int = 10000
  N: int = 10
  max_seq_len: int = 1024
