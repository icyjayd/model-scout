import numpy as np
from collections import Counter

AA20 = "ACDEFGHIKLMNPQRSTVWY"

def encode_aac(seqs):
    feats = []
    for s in seqs:
        s = str(s).upper()
        counts = Counter(s)
        feats.append([counts.get(a, 0) / len(s) for a in AA20])
    return np.array(feats, dtype=float)

def encode_kmer(seqs, k=2):
    from itertools import product
    alphabet = list(AA20)
    kmers = ["".join(p) for p in product(alphabet, repeat=k)]
    feats = []
    for s in seqs:
        s = str(s).upper()
        counts = Counter(s[i : i + k] for i in range(len(s) - k + 1))
        feats.append([counts.get(km, 0) / (len(s) - k + 1) for km in kmers])
    return np.array(feats, dtype=float)

def encode_onehot(seqs):
    aa_to_idx = {a: i for i, a in enumerate(AA20)}
    max_len = max(len(str(s)) for s in seqs)
    arr = np.zeros((len(seqs), max_len, len(AA20)), dtype=float)
    for i, s in enumerate(seqs):
        for j, a in enumerate(str(s).upper()[:max_len]):
            if a in aa_to_idx:
                arr[i, j, aa_to_idx[a]] = 1.0
    return arr.reshape(len(seqs), -1)

def encode_sequences(seqs, encoding="aac", **kwargs):
    encoding = encoding.lower()
    if encoding == "aac":
        return encode_aac(seqs)
    elif encoding == "kmer":
        return encode_kmer(seqs, k=kwargs.get("k", 2))
    elif encoding == "onehot":
        return encode_onehot(seqs)
    else:
        raise ValueError(f"Unknown encoding '{encoding}'")