"""
K-Nearest Neighbors (KNN) Classification on Car Evaluation Dataset
-----------------------------------------------------------------
Author: Navid
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load dataset
# -----------------------------
data = pd.read_csv("data/car_evaluation.csv")

# Separate features and target
X = data.drop("class", axis=1)
y = data["class"]

# -----------------------------
# 2. Encode categorical features
# -----------------------------
# Use one-hot encoding for categorical features
X_encoded = pd.get_dummies(X, columns=X.columns)

# Encode target labels to numeric
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -----------------------------
# 3. Split dataset (train/test)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# -----------------------------
# 4. Feature scaling (optional but recommended)
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5. Train KNN classifier
# -----------------------------
k = 5  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

# -----------------------------
# 6. Predict on test set
# -----------------------------
y_pred = knn.predict(X_test_scaled)

# -----------------------------
# 7. Evaluate model
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - KNN on Car Evaluation")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=300)
plt.show()

# Optional: plot accuracy for different k values
accuracies = []
k_values = range(1, 21)
for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)
    y_temp = knn_temp.predict(X_test_scaled)
    accuracies.append(accuracy_score(y_test, y_temp))

plt.figure(figsize=(7,5))
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs k")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/accuracy_plot.png", dpi=300)
plt.show()
