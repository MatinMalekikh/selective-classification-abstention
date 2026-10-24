from sklearn.datasets import load_wine
import pandas as pd

# Load the dataset
wine = load_wine(as_frame=True)

X = wine.data
y = wine.target

# Combine features and target
df = X.copy()
df["target"] = y

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nClass distribution:")
print(df["target"].value_counts())

print("\nStatistical summary:")
print(df.describe())