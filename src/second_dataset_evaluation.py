
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score


from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

print("Dataset shape:", X.shape)
print("Classes:", np.unique(y, return_counts=True))


thresholds = np.arange(0.50, 1.00, 0.01)
target_accuracy = 0.98
results = []

for seed in range(20):

    # 60% train, 20% validation, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.40,
        random_state=seed,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=seed + 100,
        stratify=y_temp
    )

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=seed
        ),
        "SVM": make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                estimator=SVC(),
                method="sigmoid",
                cv=3
            )
        )
    }

    for name, model in models.items():

        model.fit(X_train, y_train)

        # Select threshold on validation data only
        val_probs = model.predict_proba(X_val)
        val_pred = model.classes_[
            np.argmax(val_probs, axis=1)
        ]
        val_conf = val_probs.max(axis=1)

        candidates = []

        for threshold in thresholds:
            accepted = val_conf >= threshold

            if not accepted.any():
                continue

            acc = accuracy_score(
                y_val[accepted],
                val_pred[accepted]
            )

            if acc >= target_accuracy:
                candidates.append(
                    (accepted.mean(), threshold, acc)
                )

        if candidates:
            _, best_threshold, val_acc = max(
                candidates,
                key=lambda item: (item[0], -item[1])
            )
            target_met = True

        else:
            # Fallback: best validation selective accuracy,
            # breaking ties by coverage and then lower threshold.
            fallback = []

            for threshold in thresholds:
                accepted = val_conf >= threshold

                if not accepted.any():
                    continue

                acc = accuracy_score(
                    y_val[accepted],
                    val_pred[accepted]
                )

                fallback.append(
                    (acc, accepted.mean(), threshold)
                )

            val_acc, _, best_threshold = max(
                fallback,
                key=lambda item: (item[0], item[1], -item[2])
            )
            target_met = False

        # Evaluate once on the untouched test split
        test_probs = model.predict_proba(X_test)
        test_pred = model.classes_[
            np.argmax(test_probs, axis=1)
        ]
        test_conf = test_probs.max(axis=1)

        accepted = test_conf >= best_threshold

        results.append({
            "Seed": seed,
            "Model": name,
            "Threshold": best_threshold,
            "Target Met on Validation": target_met,
            "Test Accuracy": accuracy_score(
                y_test, test_pred
            ),
            "Test Macro F1": f1_score(
                y_test, test_pred, average="macro"
            ),
            "Test Coverage": accepted.mean(),
            "Test Accepted": int(accepted.sum()),
            "Test Selective Accuracy": (
                accuracy_score(
                    y_test[accepted],
                    test_pred[accepted]
                )
                if accepted.any() else np.nan
            )
        })

df = pd.DataFrame(results)
# Save Breast Cancer results in the correct folder
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (
    SCRIPT_DIR if (SCRIPT_DIR / "results").exists()
    else SCRIPT_DIR.parent
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "breast_cancer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_DIR / "repeated_evaluation_raw.csv", index=False)

summary = (
    df.groupby("Model")
    .agg(
        Mean_Test_Accuracy=("Test Accuracy", "mean"),
        Std_Test_Accuracy=("Test Accuracy", "std"),
        Mean_Macro_F1=("Test Macro F1", "mean"),
        Mean_Coverage=("Test Coverage", "mean"),
        Std_Coverage=("Test Coverage", "std"),
        Mean_Selective_Accuracy=(
            "Test Selective Accuracy", "mean"
        ),
        Successful_Validation_Targets=(
            "Target Met on Validation", "sum"
        )
    )
    .reset_index()
    .round(3)
)

summary.to_csv(OUTPUT_DIR / "repeated_evaluation_summary.csv", index=False)

print("\nREPEATED EVALUATION SUMMARY\n")
print(summary.to_string(index=False))
print("\nResults saved to CSV files.")
