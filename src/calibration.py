
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score

# 1. Load dataset
wine = load_wine()
X = wine.data
y = wine.target

# 2. Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 3. Define the base model
base_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)

# 4. Define the calibrated model
calibrated_model = CalibratedClassifierCV(
    estimator=make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000)
    ),
    method="sigmoid",
    cv=3
)

# 5. Train both models
base_model.fit(X_train, y_train)
calibrated_model.fit(X_train, y_train)

models = {
    "Original": base_model,
    "Calibrated": calibrated_model
}

results = []

# 6. Evaluate probabilities and abstention
thresholds = [0.50, 0.70, 0.90]

for name, model in models.items():

    probabilities = model.predict_proba(X_test)
    predictions = model.predict(X_test)
    confidence = probabilities.max(axis=1)

    print(f"\n{name} model")
    print("Accuracy:", round(
        accuracy_score(y_test, predictions), 3
    ))
    print("Log loss:", round(
        log_loss(y_test, probabilities), 3
    ))

    for threshold in thresholds:

        accepted = confidence >= threshold
        coverage = accepted.mean()

        if accepted.any():
            selective_accuracy = accuracy_score(
                y_test[accepted],
                predictions[accepted]
            )
        else:
            selective_accuracy = np.nan

        results.append({
            "Model": name,
            "Threshold": threshold,
            "Coverage": coverage,
            "Accepted": int(accepted.sum()),
            "Selective Accuracy": selective_accuracy
        })

    # 7. Plot confidence against actual correctness
    correct = (predictions == y_test).astype(int)
    bins = np.linspace(0.3, 1.0, 8)

    bin_centers = []
    bin_accuracies = []
    bin_counts = []

    for i in range(len(bins) - 1):
        mask = (
            (confidence >= bins[i]) &
            (confidence < bins[i + 1])
        )

        if mask.sum() > 0:
            bin_centers.append(confidence[mask].mean())
            bin_accuracies.append(correct[mask].mean())
            bin_counts.append(mask.sum())

    plt.plot(
        bin_centers,
        bin_accuracies,
        marker="o",
        label=name
    )

plt.plot(
    [0.3, 1.0],
    [0.3, 1.0],
    linestyle="--",
    label="Perfect calibration"
)

plt.xlabel("Mean Predicted Confidence")
plt.ylabel("Observed Accuracy")
plt.title("Confidence Calibration")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. Print comparison table
results_df = pd.DataFrame(results)
print("\nAbstention comparison:")
print(results_df.round(3).to_string(index=False))
