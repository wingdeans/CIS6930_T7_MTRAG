import json
import pandas as pd
import numpy as np

# === CONFIG ===
input_file = "meta-llama_Meta-Llama-3.1-8B_raft_1_cot_0.eval.jsonl"

# Automatically derive output name from input_file
base_name = os.path.splitext(os.path.basename(input_file))[0]  # e.g. meta-llama_Meta-Llama-3.1-8B_raft_1_cot_0.eval
output_csv = f"metrics_{base_name}.csv"


# --- helper to safely flatten metric values ---
def flatten_metric(value):
    """
    Converts metric values into a single float if possible.
    Handles lists of lists, singletons, or scalars.
    """
    if isinstance(value, list):
        # Recursively flatten any nested lists
        flat = []
        for v in value:
            if isinstance(v, list):
                flat.extend(v)
            else:
                flat.append(v)
        # Take the mean of numeric elements
        numeric = [x for x in flat if isinstance(x, (int, float))]
        return np.mean(numeric) if numeric else np.nan
    elif isinstance(value, (int, float)):
        return float(value)
    else:
        return np.nan


# === STEP 1: Read JSON and flatten metrics ===
records = []
with open(input_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            metrics = data.get("metrics", {})
            flat = {k: flatten_metric(v) for k, v in metrics.items()}
            records.append(flat)
        except Exception as e:
            print(f"Skipping line: {e}")

# === STEP 2: Convert to DataFrame ===
df = pd.DataFrame(records)

# === STEP 3: Add aggregated metric (robust version) ===
# Automatically choose between RB_alg or RB_agg
retrieval_col = "RB_alg" if "RB_alg" in df.columns else "RB_agg"

if all(col in df.columns for col in [retrieval_col, "RB_llm", "RL_F"]):
    print(f"✅ Using '{retrieval_col}' as retrieval metric for RB_overall.")
    df["RB_overall"] = (
        0.4 * df[retrieval_col]
        + 0.3 * df["RB_llm"]
        + 0.3 * df["RL_F"]
    )
else:
    print("⚠️ Skipping RB_overall computation: missing one of RB_alg/RB_agg, RB_llm, or RL_F")

# === STEP 4: Compute summary ===
numeric_df = df.select_dtypes(include=["number"])
summary = numeric_df.agg(["mean", "std"]).T
summary = summary.rename(columns={"mean": "Mean", "std": "StdDev"})

# === STEP 5: Print formatted table ===
print("\n| Metric | Mean | StdDev |")
print("| ------- | ---- | ------- |")
for metric, row in summary.iterrows():
    print(f"| {metric:<15} | {row['Mean']:.3f} | {row['StdDev']:.3f} |")

# === STEP 6: Save ===
summary.to_csv(output_csv, index_label="Metric")
print(f"\n✅ Metrics summary saved to {output_csv}")
