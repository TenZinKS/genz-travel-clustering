import pandas as pd
import pickle
from sklearn.cluster import KMeans

# Load processed data
df = pd.read_csv("outputs/processed_features.csv")

# Train final model with k=4
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(df)

# Save model
with open("models/kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

print("Model saved successfully")
