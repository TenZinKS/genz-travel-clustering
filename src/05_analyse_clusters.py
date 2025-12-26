import pandas as pd

# Load clustered data
df = pd.read_csv("outputs/clustered_users.csv")

print("Data loaded:", df.shape)

# Compute mean values for each cluster
cluster_summary = df.groupby("cluster").mean(numeric_only=True)

print("\nCluster Behaviour Summary:\n")
print(cluster_summary)

# Save summary
cluster_summary.to_csv("outputs/cluster_summary.csv")
print("\nSaved cluster_summary.csv to outputs/")
