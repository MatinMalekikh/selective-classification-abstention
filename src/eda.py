
from sklearn.datasets import load_wine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
wine = load_wine(as_frame=True)
X = wine.data
y = wine.target

df = X.copy()
df["target"] = y

# 1. Dataset overview
print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["target"].value_counts().sort_index())

print("\nMissing values:", df.isnull().sum().sum())

# 2. Distribution of target classes
sns.countplot(data=df, x="target")
plt.title("Distribution of Wine Classes")
plt.xlabel("Wine Class")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.show()

# 3. Distribution of selected features
selected_features = [
    "alcohol",
    "malic_acid",
    "color_intensity",
    "flavanoids"
]

df[selected_features].hist(figsize=(10, 7), bins=15)
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()

# 4. Correlation heatmap
plt.figure(figsize=(12, 9))
correlation = X.corr()

sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# 5. Compare one feature across classes
sns.boxplot(data=df, x="target", y="flavanoids")
plt.title("Flavanoids by Wine Class")
plt.xlabel("Wine Class")
plt.ylabel("Flavanoids")
plt.tight_layout()
plt.show()
