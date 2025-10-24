"""
Unified sequence encoding utilities with automatic and subset-aware caching.
Encodings: aac, dpc, kmer, onehot, esm
"""

import os
import hashlib
import json
import numpy as np
from itertools import product
from collections import Counter
from tqdm import tqdm

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)
AA_IDX = {a: i for i, a in enumerate(AA)}


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------
def _clean_sequence(seq: str) -> str:
    """Strip invalid characters and uppercase."""
    return "".join([a for a in seq.upper() if a in AA_SET])


def _hash_sequences(sequences, **kwargs):
    """Generate a short hash over first 100 sequences + params."""
    sample = "".join(sequences[:100])
    params = json.dumps(kwargs, sort_keys=True)
    key = hashlib.md5((sample + params + str(len(sequences))).encode()).hexdigest()[:10]
    return key


def _cache_path(encoding, key, base="runs/cache"):
    os.makedirs(os.path.join(base, encoding), exist_ok=True)
    return os.path.join(base, encoding, f"{encoding}_{key}.npy")


def _meta_path(encoding, key, base="runs/cache"):
    os.makedirs(os.path.join(base, encoding), exist_ok=True)
    return os.path.join(base, encoding, f"{encoding}_{key}.meta.json")


def _maybe_load_cache(encoding, key, n_request, base="runs/cache"):
    npy_path = _cache_path(encoding, key, base)
    meta_path = _meta_path(encoding, key, base)
    if not (os.path.isfile(npy_path) and os.path.isfile(meta_path)):
        return None
    try:
        meta = json.load(open(meta_path))
        total_cached = meta.get("n_sequences", 0)
        if total_cached >= n_request:
            arr = np.load(npy_path, mmap_mode="r")
            return np.array(arr[:n_request])
    except Exception as e:
        return None
    return None


def _save_cache(encoding, key, arr, base="runs/cache"):
    path = _cache_path(encoding, key, base)
    np.save(path, arr)
    meta_path = _meta_path(encoding, key, base)
    with open(meta_path, "w") as f:
        json.dump({"n_sequences": len(arr), "shape": arr.shape}, f)
    return path


# ---------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------
def encode_aac(sequences):
    vecs = np.zeros((len(sequences), len(AA)), dtype=float)
    for i, seq in enumerate(sequences):
        seq = _clean_sequence(seq)
        if not seq:
            continue
        counts = Counter(seq)
        L = len(seq)
        for a, c in counts.items():
            vecs[i, AA_IDX[a]] = c / L
    return vecs


def encode_dpc(sequences):
    dipeptides = [''.join(p) for p in product(AA, repeat=2)]
    dp_idx = {dp: i for i, dp in enumerate(dipeptides)}
    vecs = np.zeros((len(sequences), len(dipeptides)), dtype=float)
    for i, seq in enumerate(sequences):
        seq = _clean_sequence(seq)
        if len(seq) < 2:
            continue
        counts = Counter([seq[j:j + 2] for j in range(len(seq) - 1)])
        total = sum(counts.values())
        for dp, c in counts.items():
            vecs[i, dp_idx[dp]] = c / total
    return vecs


def encode_kmer(sequences, k=3):
    kmers = [''.join(p) for p in product(AA, repeat=k)]
    k_idx = {km: i for i, km in enumerate(kmers)}
    vecs = np.zeros((len(sequences), len(kmers)), dtype=float)
    for i, seq in enumerate(sequences):
        seq = _clean_sequence(seq)
        if len(seq) < k:
            continue
        counts = Counter([seq[j:j + k] for j in range(len(seq) - k + 1)])
        total = sum(counts.values())
        for km, c in counts.items():
            vecs[i, k_idx[km]] = c / total
    return vecs


def encode_onehot(sequences, max_len=512):
    n = len(sequences)
    arr = np.zeros((n, max_len, len(AA)), dtype=float)
    for i, seq in enumerate(sequences):
        seq = _clean_sequence(seq)[:max_len]
        for j, a in enumerate(seq):
            arr[i, j, AA_IDX[a]] = 1.0
    return arr.reshape(n, -1)


def encode_esm(sequences, model_name="facebook/esm2_t6_8M_UR50D", device=None):
    try:
        from transformers import EsmTokenizer, EsmModel
        import torch
    except ImportError:
        raise ImportError("Install transformers + torch to use ESM encoding.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device).eval()

    vecs = []
    for seq in tqdm(sequences, desc="ESM encoding", disable=len(sequences) < 5):
        seq = _clean_sequence(seq)
        if not seq:
            vecs.append(np.zeros(model.config.hidden_size))
            continue
        toks = tokenizer(seq, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            out = model(**toks).last_hidden_state.mean(dim=1)
        vecs.append(out.cpu().numpy().squeeze())
    return np.vstack(vecs)


# ---------------------------------------------------------------------
# Unified interface with caching + subset detection
# ---------------------------------------------------------------------
def encode_sequences(
    sequences,
    encoding="aac",
    use_cache=True,
    cache_base="runs/cache",
    **kwargs
):
    """
    Encode sequences using the selected scheme.
    Automatically caches and reuses partial results.

    Args:
        sequences (list[str]): sequences to encode
        encoding (str): aac, dpc, kmer, onehot, esm
        use_cache (bool): use caching
        cache_base (str): root cache directory
        kwargs: params for specific encoders (k, max_len, model_name, etc.)
    """
    encoding = encoding.lower()
    n_request = len(sequences)
    key = _hash_sequences(sequences, encoding=encoding, **kwargs)

    # Try to load cache
    if use_cache:
        cached = _maybe_load_cache(encoding, key, n_request, base=cache_base)
        if cached is not None:
            return cached
    # Compute new encodings
    if encoding == "aac":
        arr = encode_aac(sequences)
    elif encoding == "dpc":
        arr = encode_dpc(sequences)
    elif encoding == "kmer":
        arr = encode_kmer(sequences, k=kwargs.get("k", 3))
    elif encoding == "onehot":
        arr = encode_onehot(sequences, max_len=kwargs.get("max_len", 512))
    elif encoding == "esm":
        arr = encode_esm(
            sequences,
            model_name=kwargs.get("model_name", "facebook/esm2_t6_8M_UR50D"),
            device=kwargs.get("device"),
        )
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    # Save/update cache if allowed
    if use_cache:
        _save_cache(encoding, key, arr, base=cache_base)

    return arr
