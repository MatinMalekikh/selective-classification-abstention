
import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# 1. Load dataset
wine = load_wine()
X = wine.data
y = wine.target

# 2. Experiment settings
fractions = [0.20, 0.40, 0.60, 0.80, 1.00]
thresholds = [0.50, 0.70, 0.90]
seeds = range(20)

results = []

# 3. Repeat experiments
for seed in seeds:

    X_pool, X_test, y_pool, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y
    )

    for fraction in fractions:

        if fraction == 1.00:
            X_train = X_pool
            y_train = y_pool
        else:
            X_train, _, y_train, _ = train_test_split(
                X_pool,
                y_pool,
                train_size=fraction,
                random_state=seed,
                stratify=y_pool
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
                SVC(probability=True, random_state=seed)
            )
        }

        for name, model in models.items():

            model.fit(X_train, y_train)

            probabilities = model.predict_proba(X_test)
            predictions = model.predict(X_test)
            confidence = probabilities.max(axis=1)

            # Overall performance
            accuracy = accuracy_score(y_test, predictions)

            macro_f1 = f1_score(
                y_test,
                predictions,
                average="macro"
            )

            # Abstention performance
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
                    "Seed": seed,
                    "Model": name,
                    "Fraction": fraction,
                    "Training Samples": len(X_train),
                    "Threshold": threshold,
                    "Accuracy": accuracy,
                    "Macro F1": macro_f1,
                    "Coverage": coverage,
                    "Accepted Samples": int(accepted.sum()),
                    "Selective Accuracy": selective_accuracy
                })

# 4. Save all experiment results
df = pd.DataFrame(results)
df.to_csv("model_comparison_results.csv", index=False)

# 5. Summarize results across repeated experiments
summary = (
    df.groupby(["Model", "Fraction", "Threshold"])
    .agg(
        Training_Samples=("Training Samples", "mean"),
        Mean_Accuracy=("Accuracy", "mean"),
        Std_Accuracy=("Accuracy", "std"),
        Mean_F1=("Macro F1", "mean"),
        Mean_Coverage=("Coverage", "mean"),
        Std_Coverage=("Coverage", "std"),
        Mean_Selective_Accuracy=("Selective Accuracy", "mean"),
        Std_Selective_Accuracy=("Selective Accuracy", "std")
    )
    .reset_index()
)

summary = summary.round(3)

summary.to_csv("model_comparison_summary.csv", index=False)

print("\nEXPERIMENT SUMMARY\n")
print(summary.to_string(index=False))

print("\nSaved:")
print("- model_comparison_results.csv")
print("- model_comparison_summary.csv")
