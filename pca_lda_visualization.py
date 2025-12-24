"""
Dimensionality Reduction using PCA and LDA
-----------------------------------------
This script applies Principal Component Analysis (PCA) and
Linear Discriminant Analysis (LDA) to a labeled dataset
and visualizes the results in two dimensions.

Dataset: Iris
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------
iris = load_iris()
X = iris.data          # Feature matrix
y = iris.target        # Class labels
class_names = iris.target_names

# --------------------------------------------------
# 2. Standardize Features
# (Important for PCA performance)
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# 3. Apply PCA (2 components)
# --------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# --------------------------------------------------
# 4. Visualize PCA Result
# --------------------------------------------------
plt.figure(figsize=(7, 5))
for label in np.unique(y):
    plt.scatter(
        X_pca[y == label, 0],
        X_pca[y == label, 1],
        label=class_names[label]
    )

plt.title("PCA - 2D Projection")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 5. Apply LDA (2 components)
# --------------------------------------------------
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# --------------------------------------------------
# 6. Visualize LDA Result
# --------------------------------------------------
plt.figure(figsize=(7, 5))
for label in np.unique(y):
    plt.scatter(
        X_lda[y == label, 0],
        X_lda[y == label, 1],
        label=class_names[label]
    )

plt.title("LDA - 2D Projection")
plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
