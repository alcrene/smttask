from smttask.config import config
from . import sha1_hashing
try:
    from . import xx_hashing
except ImportError:
    xx_hashing = None

def __getattr__(name):
    match config.hash_algo:
        case "xxhash":
            return getattr(xx_hashing, name)
        case "sha1":
            return getattr(sha1_hashing, name)
        case _:
            raise RuntimeError(f"Unexpected hash algorithm '{config.hash_algo}'")  # Should never happen since Config is validated

def stablehash(o: bytes|str):
    return __getattr__("stablehash")(o)

def stablehexdigest(o: bytes|str) -> str:
    return __getattr__("stablehexdigest")(o)

def stablebytesdigest(o: bytes|str) -> bytes:
    return __getattr__("stablebytesdigest")(o)

def stableintdigest(o: bytes|str, byte_len=4) -> int:
    return __getattr__("stableintdigest")(o, byte_len=byte_len)
