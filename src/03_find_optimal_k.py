# ---------------------------------
# Step 3: Find optimal K using Elbow Method
# ---------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load processed features
df = pd.read_csv("outputs/processed_features.csv")

print("Data loaded for clustering:", df.shape)

# Range of K values to test
k_values = range(2, 11)

inertia_values = []

# Run KMeans for each K
for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans.fit(df)
    inertia_values.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia_values, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig("outputs/elbow_plot.png")
plt.show()

print("Elbow plot saved to outputs/elbow_plot.png")
