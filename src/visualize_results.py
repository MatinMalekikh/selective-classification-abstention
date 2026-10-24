
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Find project folders
SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = (
    SCRIPT_DIR
    if (SCRIPT_DIR / "results").exists()
    else SCRIPT_DIR.parent
)

RESULTS_FILE = PROJECT_ROOT / "results" / "model_comparison_summary.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load experiment summary
df = pd.read_csv(RESULTS_FILE)

models = df["Model"].unique()

# Figure 1: Accuracy vs training data size
plt.figure(figsize=(9, 5))

for model in models:
    subset = df[
        (df["Model"] == model) &
        (df["Threshold"] == 0.5)
    ].sort_values("Fraction")

    if not subset.empty:
        plt.errorbar(
            subset["Fraction"],
            subset["Mean_Accuracy"],
            yerr=subset["Std_Accuracy"],
            marker="o",
            capsize=4,
            label=model
        )

plt.xlabel("Fraction of Training Data")
plt.ylabel("Accuracy")
plt.title("Model Performance vs Training Data Size")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "accuracy_vs_data.png", dpi=200)
plt.close()

# Figure 2: Coverage vs threshold with limited training data
plt.figure(figsize=(9, 5))

for model in models:
    subset = df[
        (df["Model"] == model) &
        (df["Fraction"] == 0.2)
    ].sort_values("Threshold")

    if not subset.empty:
        plt.errorbar(
            subset["Threshold"],
            subset["Mean_Coverage"],
            yerr=subset["Std_Coverage"],
            marker="o",
            capsize=4,
            label=model
        )

plt.xlabel("Confidence Threshold")
plt.ylabel("Coverage")
plt.title("Coverage vs Threshold (20% Training Data)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "coverage_vs_threshold.png", dpi=200)
plt.close()

# Figure 3: Selective accuracy vs coverage
plt.figure(figsize=(9, 5))

for model in models:
    subset = df[
        (df["Model"] == model) &
        (df["Fraction"] == 0.2)
    ].sort_values("Threshold")

    if not subset.empty:
        plt.plot(
            subset["Mean_Coverage"],
            subset["Mean_Selective_Accuracy"],
            marker="o",
            label=model
        )

plt.xlabel("Coverage")
plt.ylabel("Selective Accuracy")
plt.title("Selective Accuracy vs Coverage (20% Training Data)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "selective_accuracy_vs_coverage.png",
    dpi=200
)
plt.close()

print(f"Figures saved to: {FIGURES_DIR}")
