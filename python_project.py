print("Machine Learning projekat inicijalizovan.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
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

print("\nLogistic Regression results:")

# Create and train the model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Make predictions
y_pred = logreg.predict(X_test)
y_proba = logreg.predict_proba(X_test)[:, 1]

# Display evaluation metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_proba))

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Disease", "Disease"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# Initialize and train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

# Evaluation metrics
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))
print("AUC-ROC:", roc_auc_score(y_test, y_proba_rf))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=["No Disease", "Disease"]
)

disp_rf.plot(cmap="Greens")
plt.title("Confusion Matrix - Random Forest")
plt.show()

print("\nK-Nearest Neighbors results:")

# Create and train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred_knn = knn.predict(X_test)
y_proba_knn = knn.predict_proba(X_test)[:, 1]

# Display evaluation metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn, zero_division=0))

print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print(
    "Precision:",
    precision_score(y_test, y_pred_knn, zero_division=0)
)
print(
    "Recall:",
    recall_score(y_test, y_pred_knn, zero_division=0)
)
print(
    "F1 Score:",
    f1_score(y_test, y_pred_knn, zero_division=0)
)
print("AUC-ROC:", roc_auc_score(y_test, y_proba_knn))

# Display confusion matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)

disp_knn = ConfusionMatrixDisplay(
    confusion_matrix=cm_knn,
    display_labels=["No Disease", "Disease"]
)

disp_knn.plot(cmap="Oranges")
plt.title("Confusion Matrix - KNN")
plt.show()

print("\nDecision Tree results:")

# Create and train the Decision Tree model
decision_tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

decision_tree.fit(X_train, y_train)

# Make predictions
y_pred_dt = decision_tree.predict(X_test)
y_proba_dt = decision_tree.predict_proba(X_test)[:, 1]

# Display evaluation metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt, zero_division=0))

print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(
    "Precision:",
    precision_score(y_test, y_pred_dt, zero_division=0)
)
print(
    "Recall:",
    recall_score(y_test, y_pred_dt, zero_division=0)
)
print(
    "F1 Score:",
    f1_score(y_test, y_pred_dt, zero_division=0)
)
print("AUC-ROC:", roc_auc_score(y_test, y_proba_dt))

# Display confusion matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)

disp_dt = ConfusionMatrixDisplay(
    confusion_matrix=cm_dt,
    display_labels=["No Disease", "Disease"]
)

disp_dt.plot(cmap="Purples")
plt.title("Confusion Matrix - Decision Tree")
plt.show()

print("\nGenerating learning curves:")

models_for_learning_curves = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),
    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

training_sizes = np.linspace(0.1, 1.0, 5)

for model_name, model_for_curve in models_for_learning_curves.items():
    sizes, training_scores, validation_scores = learning_curve(
        model_for_curve,
        X,
        y,
        train_sizes=training_sizes,
        cv=5,
        scoring="accuracy"
    )

    training_mean = np.mean(training_scores, axis=1)
    validation_mean = np.mean(validation_scores, axis=1)

    plt.figure(figsize=(7, 5))
    plt.plot(
        sizes,
        training_mean,
        marker="o",
        label="Training accuracy"
    )
    plt.plot(
        sizes,
        validation_mean,
        marker="o",
        label="Validation accuracy"
    )

    plt.title(f"Learning Curve - {model_name}")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("\nFeature importance analysis:")

# Logistic Regression coefficients
logreg_coefficients = pd.Series(
    logreg.coef_[0],
    index=X.columns
)

logreg_coefficients = logreg_coefficients.reindex(
    logreg_coefficients.abs().sort_values(ascending=False).index
)

print("\nLogistic Regression coefficients:")
print(logreg_coefficients)

plt.figure(figsize=(8, 5))
sns.barplot(
    x=logreg_coefficients.values,
    y=logreg_coefficients.index
)
plt.title("Feature Coefficients - Logistic Regression")
plt.xlabel("Coefficient")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Random Forest feature importance
rf_importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nRandom Forest feature importance:")
print(rf_importance)

plt.figure(figsize=(8, 5))
sns.barplot(
    x=rf_importance.values,
    y=rf_importance.index
)
plt.title("Feature Importances - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

print("\nFinal model comparison:")

model_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "KNN",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, y_pred_rf)
    ],
    "Precision": [
        precision_score(y_test, y_pred, zero_division=0),
        precision_score(y_test, y_pred_knn, zero_division=0),
        precision_score(y_test, y_pred_dt, zero_division=0),
        precision_score(y_test, y_pred_rf, zero_division=0)
    ],
    "Recall": [
        recall_score(y_test, y_pred, zero_division=0),
        recall_score(y_test, y_pred_knn, zero_division=0),
        recall_score(y_test, y_pred_dt, zero_division=0),
        recall_score(y_test, y_pred_rf, zero_division=0)
    ],
    "F1 Score": [
        f1_score(y_test, y_pred, zero_division=0),
        f1_score(y_test, y_pred_knn, zero_division=0),
        f1_score(y_test, y_pred_dt, zero_division=0),
        f1_score(y_test, y_pred_rf, zero_division=0)
    ],
    "AUC-ROC": [
        roc_auc_score(y_test, y_proba),
        roc_auc_score(y_test, y_proba_knn),
        roc_auc_score(y_test, y_proba_dt),
        roc_auc_score(y_test, y_proba_rf)
    ]
})

print(model_comparison.round(3))

best_model_index = model_comparison["F1 Score"].idxmax()
best_model_name = model_comparison.loc[
    best_model_index,
    "Model"
]

print("\nBest model based on F1 score:", best_model_name)

comparison_plot = model_comparison.set_index("Model")

comparison_plot.plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0.0, 1.0)
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()