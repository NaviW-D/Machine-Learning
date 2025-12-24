"""
PCA and LDA on a Kaggle Dataset
------------------------------
This script applies PCA and LDA to a labeled dataset
loaded from a CSV file and visualizes the results in 2D.
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
# NOTE: Last column must be the class label
data = pd.read_csv("data/dataset.csv")

X = data.iloc[:, :-1].values   # Features
y = data.iloc[:, -1].values    # Labels

# --------------------------------------------------
# 2. Feature Scaling
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# 3. PCA (2 components)
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
        label=str(label)
    )

plt.title("PCA - 2D Projection (Kaggle Dataset)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 5. LDA (2 components)
# --------------------------------------------------
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# --------------------------------------------------
# 6. LDA Visualization
# --------------------------------------------------
plt.figure(figsize=(7, 5))
for label in np.unique(y):
    plt.scatter(
        X_lda[y == label, 0],
        X_lda[y == label, 1],
        label=str(label)
    )

plt.title("LDA - 2D Projection (Kaggle Dataset)")
plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
