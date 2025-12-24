"""
Outlier Detection using K-Means Clustering
------------------------------------------
This script applies K-Means clustering to a dataset,
calculates the distance of each point from its cluster centroid,
detects outliers, and visualizes the results.

Author: Navid
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------
# Example: Pima Indians Diabetes dataset from Kaggle
# https://www.kaggle.com/uciml/pima-indians-diabetes-database
data = pd.read_csv("data/dataset.csv")

X = data.values  # Features only, make sure labels are excluded if any

# --------------------------------------------------
# 2. Feature Scaling
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# 3. Apply K-Means Clustering
# --------------------------------------------------
k = 3  # Number of clusters, can be adjusted
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_

# --------------------------------------------------
# 4. Calculate distance of each point to its centroid
# --------------------------------------------------
distances = np.linalg.norm(X_scaled - centroids[labels], axis=1)

# --------------------------------------------------
# 5. Detect Outliers (distance > mean + 2*std)
# --------------------------------------------------
threshold = np.mean(distances) + 2 * np.std(distances)
outliers = np.where(distances > threshold)[0]

print("Indices of outliers:", outliers)
print("Outlier points:\n", X[outliers])

# --------------------------------------------------
# 6. Visualization
# --------------------------------------------------
plt.figure(figsize=(7, 5))
for i in range(k):
    plt.scatter(
        X_scaled[labels == i, 0],
        X_scaled[labels == i, 1],
        label=f"Cluster {i}",
        alpha=0.6
    )

# Highlight outliers
plt.scatter(
    X_scaled[outliers, 0],
    X_scaled[outliers, 1],
    color='red',
    marker='x',
    s=100,
    label='Outliers'
)

plt.title("K-Means Clustering with Outliers Highlighted")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/outliers.png", dpi=300)
plt.show()
