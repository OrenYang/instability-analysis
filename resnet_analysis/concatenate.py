import pandas as pd

files = [
    "nn_training_labels_normalized_1.csv",
    "nn_training_labels_normalized_2.csv",
    "nn_training_labels_normalized_6.csv",
]

dfs = [pd.read_csv(f) for f in files]
df_all = pd.concat(dfs, ignore_index=True)

df_all.to_csv("nn_training_labels_normalized.csv", index=False)
