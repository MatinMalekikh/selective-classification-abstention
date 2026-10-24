
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (
    SCRIPT_DIR
    if (SCRIPT_DIR / "results").exists()
    else SCRIPT_DIR.parent
)

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "Wine": RESULTS_DIR / "wine" / "repeated_evaluation_raw.csv",
    "Breast Cancer": (
        RESULTS_DIR / "breast_cancer" / "repeated_evaluation_raw.csv"
    ),
}

required_columns = [
    "Seed",
    "Model",
    "Test Accuracy",
    "Test Macro F1",
    "Test Coverage",
    "Test Selective Accuracy",
]

frames = []

for dataset_name, file_path in DATASETS.items():
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing columns: {missing}"
        )

    df["Dataset"] = dataset_name
    frames.append(df)

raw = pd.concat(frames, ignore_index=True)

# Calculate mean and standard deviation across runs
metrics = [
    "Test Accuracy",
    "Test Macro F1",
    "Test Coverage",
    "Test Selective Accuracy",
]

summary = (
    raw.groupby(["Dataset", "Model"])[metrics]
    .agg(["mean", "std"])
    .round(4)
)

summary.columns = [
    f"{metric.replace('Test ', '').replace(' ', '_')}_{stat}"
    for metric, stat in summary.columns
]
summary = summary.reset_index()

summary_path = RESULTS_DIR / "final_raw_results_summary.csv"
summary.to_csv(summary_path, index=False)
print(f"Summary saved to: {summary_path}")

# Create one chart per dataset
plot_metrics = {
    "Test Accuracy": "Baseline Accuracy",
    "Test Selective Accuracy": "Selective Accuracy",
    "Test Coverage": "Coverage",
}

for dataset_name in DATASETS:
    subset = raw[raw["Dataset"] == dataset_name]

    means = subset.groupby("Model")[
        list(plot_metrics.keys())
    ].mean().rename(columns=plot_metrics)

    ax = (means * 100).plot(
        kind="bar",
        figsize=(10, 6),
        ylim=(0, 105),
        rot=0,
    )

    ax.set_title(f"Baseline vs Abstention — {dataset_name}")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Model")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")

    # Label each bar with its percentage
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.1f%%",
            padding=2,
            fontsize=8,
        )

    plt.tight_layout()


    if dataset_name == "Wine":
        filename = "baseline_vs_abstention_wine.png"
    else:
        filename = "baseline_vs_abstention_breast_cancer.png"

    output_path = FIGURES_DIR / filename
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        plt.savefig(str(output_path), dpi=200, bbox_inches="tight")
        print(f"Chart saved to: {output_path}")
    finally:
        plt.close()

print("\nAnalysis completed.")

