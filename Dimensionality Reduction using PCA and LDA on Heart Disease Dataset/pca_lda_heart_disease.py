"""
Dimensionality Reduction using PCA and LDA
------------------------------------------
Dataset: Heart Disease (Kaggle)

This script applies PCA and LDA to the Heart Disease dataset,
reduces the dimensionality, and visualizes the results.

Author: Navid
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------
# The last column is assumed to be the class label
data = pd.read_csv("data/heart_disease.csv")

X = data.iloc[:, :-1].values   # Features
y = data.iloc[:, -1].values    # Labels

# --------------------------------------------------
# 2. Feature Scaling
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# 3. Apply PCA (2 Components)
# --------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# --------------------------------------------------
# 4. PCA Visualization
# --------------------------------------------------
plt.figure(figsize=(7, 5))
for label in np.unique(y):
    plt.scatter(
        X_pca[y == label, 0],
        X_pca[y == label, 1],
        label=f"Class {label}",
        alpha=0.7
    )

plt.title("PCA - 2D Projection (Heart Disease Dataset)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/pca_heart_disease.png", dpi=300)
plt.show()

# --------------------------------------------------
# 5. Apply LDA
# For binary classification: max components = 1
# --------------------------------------------------
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X_scaled, y)

# --------------------------------------------------
# 6. LDA Visualization (1D Projection)
# --------------------------------------------------
plt.figure(figsize=(7, 3))
for label in np.unique(y):
    plt.scatter(
        X_lda[y == label],
        np.zeros_like(X_lda[y == label]),
        label=f"Class {label}",
        alpha=0.7
    )

plt.title("LDA - 1D Projection (Heart Disease Dataset)")
plt.xlabel("Linear Discriminant 1")
plt.yticks([])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/lda_heart_disease.png", dpi=300)
plt.show()
