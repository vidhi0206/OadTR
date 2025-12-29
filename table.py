import os
import re
import json
import pandas as pd
import numpy as np

# Root logs directory
root_dir = "models"

results = []

for dirpath, _, filenames in os.walk(root_dir):
    if "log_tran&test.txt" not in filenames:
        continue

    file_path = os.path.join(dirpath, "log_tran&test.txt")

    try:
        # -----------------------------
        # Path-based metadata extraction
        # -----------------------------
        layers_match = re.search(r"en_(\d+)", file_path)
        dataset_match = re.search(r"models/en_\d+/([^/]+)/", file_path)
        epoch_match = re.search(r"/(\d+)_dr_", file_path)

        dr_mha_match = re.search(r"dr_([A-Z])", file_path)
        drop_mha_match = re.search(r"drop_([^_]+)", file_path)
        mlp_match = re.search(r"mlp_(\d+)", file_path)
        dr_ratio_match = re.search(r"dr_ratio_([0-9]*\.?[0-9]+)", file_path)
        lambda_dr_match = re.search(r"lambda_dr_([0-9]*\.?[0-9]+)", file_path)
        gamma_match = re.search(r"Gamma_([0-9]*\.?[0-9]+)", file_path)

        layers = layers_match.group(1) if layers_match else None
        dataset = dataset_match.group(1) if dataset_match else None
        epochs = epoch_match.group(1) if epoch_match else None
        dr_mha = dr_mha_match.group(1) if dr_mha_match else None
        drop_mha = drop_mha_match.group(1) if drop_mha_match else None
        dr_mlp = mlp_match.group(1) if mlp_match else None
        dr_ratio = dr_ratio_match.group(1) if dr_ratio_match else None
        lambda_dr = lambda_dr_match.group(1) if lambda_dr_match else None
        gamma = gamma_match.group(1) if gamma_match else None

        # -----------------------------
        # Read JSON-lines log
        # -----------------------------
        with open(file_path, "r") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        if not lines:
            continue

        last = lines[-1]

        results.append({
            "dataset": dataset,
            "layers": layers,
            "epochs": epochs,
            "dr_mha": dr_mha,
            "drop_mha": drop_mha,
            "dr_mlp": dr_mlp,
            "dr_ratio": dr_ratio,
            "lambda_dr": lambda_dr,
            "gamma": gamma,
            "train_loss": last.get("train_loss"),
            "test_loss": last.get("test_loss"),
            "test_map": last.get("test_map"),
            #"n_parameters": last.get("n_parameters"),
        })

    except Exception as e:
        print(f"Skipping {file_path}: {e}")

# -----------------------------
# Sort by best MAP
# -----------------------------
results.sort(
    key=lambda x: (
        -1 if x["test_map"] is None or np.isnan(x["test_map"]) else x["test_map"]
    ),
    reverse=True,
)

df_results = pd.DataFrame(results)

# -----------------------------
# LaTeX table
# -----------------------------
latex_table = df_results.to_latex(
    index=False,
    float_format="{:.4f}".format
)

latex_table = (
    latex_table
    .replace("NaN", "-")
    .replace("_", "\\_")
    .replace("tabular", "longtable")
)

with open("results_table.tex", "w") as f:
    f.write(latex_table)

print("LaTeX table saved as results_table.tex")
