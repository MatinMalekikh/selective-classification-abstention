
import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------
# 1. Load data
# ---------------------------------
wine = load_wine()
X = wine.data
y = wine.target

# 60% train, 20% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

# ---------------------------------
# 2. Define models
# ---------------------------------
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000)
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
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

# Candidate thresholds
thresholds = np.arange(0.50, 1.00, 0.01)
target_accuracy = 0.98

results = []

# ---------------------------------
# 3. Train and select threshold
# ---------------------------------
for name, model in models.items():

    model.fit(X_train, y_train)

    # Validation: select threshold only here
    val_probs = model.predict_proba(X_val)
    val_pred = model.classes_[np.argmax(val_probs, axis=1)]
    val_conf = val_probs.max(axis=1)

    candidates = []

    for threshold in thresholds:
        accepted = val_conf >= threshold

        if accepted.sum() == 0:
            continue

        coverage = accepted.mean()
        selective_acc = accuracy_score(
            y_val[accepted],
            val_pred[accepted]
        )

        if selective_acc >= target_accuracy:
            candidates.append(
                (coverage, threshold, selective_acc)
            )

    if candidates:
        # Maximum coverage; use lower threshold to break ties
        best_coverage, best_threshold, val_selective_acc = max(
            candidates,
            key=lambda item: (item[0], -item[1])
        )
        meets_target = True
    else:
        # No threshold meets target: select best validation accuracy
        fallback = []

        for threshold in thresholds:
            accepted = val_conf >= threshold

            if accepted.sum() == 0:
                continue

            acc = accuracy_score(
                y_val[accepted],
                val_pred[accepted]
            )

            fallback.append(
                (acc, accepted.mean(), threshold)
            )

        best_acc, best_coverage, best_threshold = max(
            fallback,
            key=lambda item: (item[0], item[1], -item[2])
        )

        val_selective_acc = best_acc
        meets_target = False

    # ---------------------------------
    # 4. Final evaluation on untouched test
    # ---------------------------------
    test_probs = model.predict_proba(X_test)
    test_pred = model.classes_[np.argmax(test_probs, axis=1)]
    test_conf = test_probs.max(axis=1)

    accepted = test_conf >= best_threshold

    test_coverage = accepted.mean()

    if accepted.any():
        test_selective_acc = accuracy_score(
            y_test[accepted],
            test_pred[accepted]
        )
    else:
        test_selective_acc = np.nan

    results.append({
        "Model": name,
        "Threshold": best_threshold,
        "Validation Coverage": best_coverage,
        "Validation Selective Accuracy": val_selective_acc,
        "Target Met on Validation": meets_target,
        "Test Accuracy": accuracy_score(y_test, test_pred),
        "Test Macro F1": f1_score(
            y_test, test_pred, average="macro"
        ),
        "Test Coverage": test_coverage,
        "Test Accepted Samples": int(accepted.sum()),
        "Test Selective Accuracy": test_selective_acc
    })

# ---------------------------------
# 5. Display and save results
# ---------------------------------
results_df = pd.DataFrame(results).round(3)

print("\nFINAL TEST RESULTS\n")
print(results_df.to_string(index=False))

results_df.to_csv(
    "selective_evaluation_results.csv",
    index=False
)

print("\nSaved: selective_evaluation_results.csv")
