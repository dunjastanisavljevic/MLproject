print("Machine Learning projekat inicijalizovan.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv(
    "data/HeartDiseaseRiskPredictionDataset/heart_disease_prediction.csv"
)

print("First five rows of the dataset:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values in each column:")
print(df.isnull().sum())


print("\nEncoding categorical variables:")

df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["diabetes"] = df["diabetes"].map({"Yes": 1, "No": 0})

print(df.head())

print("\nDataset information after encoding:")
df.info()

plt.figure(figsize=(5, 4))
sns.countplot(x="heart_disease", data=df)
plt.title("Heart Disease Distribution (0 = No, 1 = Yes)")
plt.xlabel("Heart Disease")
plt.ylabel("Count")
plt.show()

num_features = ["age", "cholesterol", "bp"]

for col in num_features:
    plt.figure(figsize=(6, 4))
    sns.histplot(
        data=df,
        x=col,
        hue="heart_disease",
        kde=True,
        bins=30
    )
    plt.title(f"{col.capitalize()} Distribution by Heart Disease")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

cat_features = ["sex", "diabetes"]

for col in cat_features:
    plt.figure(figsize=(5, 4))
    sns.barplot(x=col, y="heart_disease", data=df)
    plt.title(f"Average Heart Disease Rate by {col}")
    plt.ylabel("Mean Heart Disease (1 = Yes)")
    plt.show()

print("\nCorrelation matrix:")
print(df.corr())

plt.figure(figsize=(8, 6))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("\nPreparing data for model training:")

# Separate input features and target
X = df.drop("heart_disease", axis=1)
y = df["heart_disease"]

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)