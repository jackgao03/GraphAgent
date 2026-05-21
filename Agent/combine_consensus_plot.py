import os
import json
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

sns.set_theme(style="white")


def load_data_from_json(report_dir="detailed_reports"):

    json_files = glob.glob(os.path.join(report_dir, "*.json"))
    if not json_files:
        raise FileNotFoundError(f" don't  find JSON from {report_dir}")

    records_dim1 = []
    records_dim2 = []

    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        industry_name = data.get("Industry_Name", os.path.basename(file_path))
        models_details = data.get("Expert_Models_Details", {})

        dim1_row = {"Industry": industry_name}
        dim2_row = {"Industry": industry_name}

        for model_name, details in models_details.items():
            correction_res = details.get("Expert_Correction_Process", {})
            dim1_row[model_name] = correction_res.get("core_dimension_1", "NA").strip()
            dim2_row[model_name] = correction_res.get("core_dimension_2", "NA").strip()

        records_dim1.append(dim1_row)
        records_dim2.append(dim2_row)

    df_dim1 = pd.DataFrame(records_dim1).set_index("Industry")
    df_dim2 = pd.DataFrame(records_dim2).set_index("Industry")

    return df_dim1, df_dim2

def get_consensus_counts(df):
    num_models = len(df.columns)
    consensus_levels = []

    if num_models == 3:
        order = ["3/3 Full Agreement", "2/3 Majority Agreement", "Disagreement"]
    elif num_models == 4:
        order = ["4/4 Full Agreement", "3/4 Majority Agreement", "Disagreement (<=2)"]
    else: 
        order = [
            f"{num_models}/{num_models} Full Agreement",
            f"{num_models-1}/{num_models} Strong Consensus",
            f"{num_models-2}/{num_models} Majority Agreement",
            "Disagreement"
        ]

    for _, row in df.iterrows():
        counts = Counter(row.values)
        top_count = counts.most_common(1)[0][1]

        if top_count == num_models:
            consensus_levels.append(order[0])
        elif top_count == num_models - 1:
            consensus_levels.append(order[1])
        elif num_models >= 5 and top_count == num_models - 2:
            consensus_levels.append(order[2])
        else:
            consensus_levels.append(order[-1])

    level_counts = Counter(consensus_levels)
    counts = [level_counts.get(k, 0) for k in order]

    return counts, order


def plot_combined_consensus_distribution(df_dim1, df_dim2, filename):
    counts1, order1 = get_consensus_counts(df_dim1)
    counts2, order2 = get_consensus_counts(df_dim2)

    if order1 != order2:
        raise ValueError("ERROR")

    order = order1
    x = np.arange(len(order))
    width = 0.36

    plt.figure(figsize=(12, 7))

    bars1 = plt.bar(
        x - width / 2,
        counts1,
        width,
        label="Core Dimension 1 (Driver Type)",
        color="#4C78A8",
        edgecolor="gray",
        linewidth=0.8
    )

    bars2 = plt.bar(
        x + width / 2,
        counts2,
        width,
        label="Core Dimension 2 (Inventory Cycle)",
        color="#72B7B2",
        edgecolor="gray",
        linewidth=0.8
    )

    for bar in bars1:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.3,
            int(yval),
            ha='center',
            va='bottom',
            fontsize=10
        )

    for bar in bars2:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.3,
            int(yval),
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.xticks(x, order, rotation=20, ha='right')
    plt.ylabel("Industry Number", fontsize=10)
    plt.title("Consensus Distribution Across Two Core Dimensions", fontsize=12)
    plt.legend(frameon=False)
    plt.grid(axis='y', linestyle='--', alpha=0.35)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"  Saved {filename}")


def main():
    print(">>> loading data from detailed_reports...")
    try:
        df_dim1, df_dim2 = load_data_from_json("detailed_reports")
    except FileNotFoundError as e:
        print(e)
        return


    target_models = [
        "claude",
        "gpt",
        "kimi"
    ]

    available_models = df_dim1.columns.tolist()
    valid_targets = [m for m in target_models if m in available_models]

    if len(valid_targets) < len(target_models):
        print(f"warning: can't find model in the available models: {available_models}")

    if len(valid_targets) < 2:
        print("errer: model number .")
        return


    df_dim1 = df_dim1[valid_targets]
    df_dim2 = df_dim2[valid_targets]

    plot_combined_consensus_distribution(
        df_dim1,
        df_dim2,
        "figure3_combined_consensus.png"
    )


if __name__ == "__main__":
    main()