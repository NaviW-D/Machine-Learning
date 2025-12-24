"""
DBSCAN Clustering and Epsilon Selection
Author: Navid
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import os

# -----------------------------
# 1. Create figures folder
# -----------------------------
if not os.path.exists("figures"):
    os.makedirs("figures")

# -----------------------------
# 2. Generate synthetic dataset
# -----------------------------
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# -----------------------------
# 3. k-distance plot
# -----------------------------
k = 4  # min_samples
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)

distances = np.sort(distances[:, k-1])
plt.figure(figsize=(6,4))
plt.plot(distances)
plt.xlabel("Points sorted by distance")
plt.ylabel(f"{k}-th nearest neighbor distance")
plt.title("k-distance plot for DBSCAN")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/k_distance_plot.png", dpi=300)
plt.show()

# -----------------------------
# 4. Select epsilon based on elbow
# -----------------------------
epsilon = 0.15
min_samples = 4

# -----------------------------
# 5. Apply DBSCAN
# -----------------------------
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
labels = dbscan.fit_predict(X)

# -----------------------------
# 6. Visualize clusters
# -----------------------------
plt.figure(figsize=(6,5))
plt.scatter(X[:,0], X[:,1], c=labels, cmap='Paired', s=50)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title(f"DBSCAN clustering (ε={epsilon}, min_samples={min_samples})")
plt.tight_layout()
plt.savefig("figures/dbscan_clusters.png", dpi=300)
plt.show()

# -----------------------------
# 7. Optional: print cluster info
# -----------------------------
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"Estimated number of clusters: {n_clusters}")
print(f"Estimated number of noise points: {n_noise}")
