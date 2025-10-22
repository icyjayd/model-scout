import numpy as np
import pandas as pd

def load_data(seq_file: str, label_file: str) -> pd.DataFrame:
    if seq_file.lower().endswith(".npy"):
        seqs = np.load(seq_file, allow_pickle=True)
        df = pd.DataFrame({"sequence": seqs})
    elif seq_file.lower().endswith(".csv"):
        df = pd.read_csv(seq_file)
        if "sequence" not in df.columns:
            first = df.columns[0]
            df = pd.DataFrame({"sequence": df[first].astype(str)})
    else:
        raise ValueError("seq_file must be .csv or .npy")

    if label_file.lower().endswith(".npy"):
        labels = np.load(label_file, allow_pickle=True)
        df["label"] = labels
    elif label_file.lower().endswith(".csv"):
        lbl = pd.read_csv(label_file)
        lbl = lbl.loc[:, ~lbl.columns.str.contains(r"^Unnamed")]
        if "label" not in lbl.columns:
            lbl.columns = ["label"]
        df["label"] = lbl.iloc[:, 0].values
    else:
        raise ValueError("label_file must be .csv or .npy")

    return df