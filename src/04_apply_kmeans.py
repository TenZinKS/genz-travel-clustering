import pandas as pd
from sklearn.cluster import KMeans

# Load processed features
df = pd.read_csv("outputs/processed_features.csv")

print("Data loaded:", df.shape)

# Apply KMeans with K = 4
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(df)

# Add cluster labels to dataframe
df["cluster"] = clusters

print("Clustering complete")
print(df["cluster"].value_counts())

# Save clustered data
df.to_csv("outputs/clustered_users.csv", index=False)
print("Saved clustered_users.csv to outputs/")
