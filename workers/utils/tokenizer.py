"""The tokenizer the chunkers count tokens with."""
from functools import lru_cache

import tiktoken

from workers.config import Config


@lru_cache
def chunk_tokenizer() -> tiktoken.Encoding:
    """Give Chonkie a tiktoken *object*, not a name: Chonkie 1.6 resolves names by downloading them from
    HuggingFace, which fails for tiktoken-only encodings like o200k_harmony (and when offline)."""
    return tiktoken.get_encoding(Config.CHONKIE_TOKENIZER)
