import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

AA20 = set("ACDEFGHIKLMNPQRSTVWY")

def normalize_sequence(seq: str) -> str:
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", str(seq).upper())

def _to_seq_list(arr: np.ndarray):
    arr = np.asarray(arr)
    if hasattr(arr, "dtype") and getattr(arr.dtype, "names", None):
        names = list(arr.dtype.names)
        if "sequence" in names:
            return [str(x) for x in arr["sequence"].tolist()]
        if "seq" in names:
            return [str(x) for x in arr["seq"].tolist()]
    if arr.ndim == 1:
        return [str(x) for x in arr.tolist()]
    if arr.ndim >= 2 and arr.shape[1] >= 1:
        return [str(x) for x in arr[:, 0].tolist()]
    raise ValueError("Unrecognized NPY sequence array shape")

def load_data(sequences_file, labels_file=None, **kwargs):
    sequences_file = str(sequences_file)
    labels_file = str(labels_file) if labels_file else None

    if sequences_file.lower().endswith(".npy"):
        seqs = np.load(sequences_file, allow_pickle=True)
        seq_list = _to_seq_list(seqs)
        df = pd.DataFrame({"sequence": seq_list})

        if labels_file:
            if labels_file.lower().endswith(".npy"):
                y = np.load(labels_file, allow_pickle=True)
                y = np.asarray(y).squeeze()
                if len(y) != len(df):
                    raise ValueError("labels_file length does not match sequences length")
                df["label"] = pd.Series(y).reset_index(drop=True)
            else:
                df_labels = pd.read_csv(labels_file)
                # drop unnamed filler columns
                df_labels = df_labels.loc[:, ~df_labels.columns.str.contains("^Unnamed")]
                if "label" not in df_labels.columns:
                    df_labels.columns = ["label"]
                if "label" in df_labels.columns and len(df_labels) == len(df):
                    df["label"] = df_labels["label"].reset_index(drop=True)
                else:
                    raise ValueError("For NPY sequences, CSV labels must have only 'label' and match length.")
        else:
            raise ValueError("Must include labels when sequences are provided as NPY. Provide labels_file.")

        df["sequence"] = df["sequence"].map(normalize_sequence)
        return df[["sequence", "label"]]

    if sequences_file.lower().endswith(".csv"):
        df = pd.read_csv(sequences_file)
        if "sequence" not in df.columns:
            # fallback: first column is the sequence
            first_col = df.columns[0]
            df = pd.DataFrame({"sequence": df[first_col].astype(str)})
        if labels_file:
            if labels_file.lower().endswith(".npy"):
                y = np.load(labels_file, allow_pickle=True)
                y = np.asarray(y).squeeze()
                if len(y) != len(df):
                    raise ValueError("labels_file length does not match sequences length")
                df["label"] = pd.Series(y).reset_index(drop=True)
            else:
                df_labels = pd.read_csv(labels_file, sep=None, engine="python")
                if "id" in df.columns and "id" in df_labels.columns:
                    df = pd.merge(df, df_labels, on="id", how="inner")
                else:
                    if "label" not in df_labels.columns and df_labels.shape[1] == 1:
                        df_labels.columns = ["label"]
                    if "label" in df_labels.columns and len(df_labels) == len(df):
                        df["label"] = df_labels["label"].reset_index(drop=True)
                    else:
                        raise ValueError("labels_file must have 'id' to merge or be same-length CSV with 'label'.")
        if "sequence" not in df.columns or "label" not in df.columns:
            raise ValueError("Must include 'sequence' and 'label'.")

        df["sequence"] = df["sequence"].map(normalize_sequence)
        return df[["sequence", "label"]]

    raise ValueError("Unsupported file format.")

def split_data(df, seed=42, test_size=0.2, task="regression"):
    strat = df["label"] if (task == "classification" and df["label"].nunique() > 1) else None
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)
