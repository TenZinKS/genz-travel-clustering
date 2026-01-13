import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load dataset
DATA_PATH = "data/genz_social_media_engagement_travel_10000.csv"
df = pd.read_csv(DATA_PATH)

print("Original shape:", df.shape)

# -----------------------------
# 1. Select behavioural features
# -----------------------------
behavioural_columns = [
    "daily_usage_hours",
    "usage_frequency",
    "trust_in_social_media",
    "influencer_influence",
    "decision_influence_level",
    "self_reported_travel_interest",
    "feels_addicted",
    "follows_travel_content",
    "engages_with_travel_posts",
    "saves_travel_content",
    "plans_trips_via_social_media",
    "primary_platform",
    "content_consumption_style",
    "visual_content_preference"
]

df_behaviour = df[behavioural_columns].copy()

print("Behavioural data shape:", df_behaviour.shape)

# -----------------------------
# 2. Convert Yes/No to 0/1
# -----------------------------
binary_columns = [
    "feels_addicted",
    "follows_travel_content",
    "saves_travel_content",
    "plans_trips_via_social_media"
]

for col in binary_columns:
    df_behaviour[col] = df_behaviour[col].map({"Yes": 1, "No": 0})

# -----------------------------
# 2.5 Convert usage_frequency to numeric
# -----------------------------
usage_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

df_behaviour["usage_frequency"] = df_behaviour["usage_frequency"].map(usage_map)

# -----------------------------
# 2.6 Convert daily_usage_hours to numeric
# -----------------------------
usage_hours_map = {
    "<1": 1,
    "1-2": 2,
    "3-5": 3,
    "5+": 4
}

df_behaviour["daily_usage_hours"] = df_behaviour["daily_usage_hours"].map(usage_hours_map)

# -----------------------------
# 2.65 Convert influencer_influence to numeric
# -----------------------------
influencer_map = {
    "No": 1,
    "Sometimes": 2,
    "Yes": 3
}

df_behaviour["influencer_influence"] = df_behaviour["influencer_influence"].map(influencer_map)

# -----------------------------
# 2.66 Convert engages_with_travel_posts to numeric
# -----------------------------
engagement_map = {
    "Rarely": 1,
    "Sometimes": 2,
    "Often": 3
}

df_behaviour["engages_with_travel_posts"] = df_behaviour["engages_with_travel_posts"].map(engagement_map)

# -----------------------------
# 2.7 Convert Low / Medium / High scales to numeric
# -----------------------------
likert_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

likert_columns = [
    "trust_in_social_media",
    "decision_influence_level",
    "self_reported_travel_interest"
]

for col in likert_columns:
    df_behaviour[col] = df_behaviour[col].map(likert_map)


# -----------------------------
# 3. One-hot encode categorical columns
# -----------------------------
categorical_columns = [
    "primary_platform",
    "content_consumption_style",
    "visual_content_preference"
]

df_encoded = pd.get_dummies(
    df_behaviour,
    columns=categorical_columns,
    drop_first=True
)

print("After encoding shape:", df_encoded.shape)

# -----------------------------
# 4. Feature scaling
# -----------------------------
scaler = StandardScaler()
# -----------------------------
# 2.8 Handle missing values (NaN)
# -----------------------------

# Check how many NaN values exist
print("Total NaN values before cleaning:", df_encoded.isna().sum().sum())

# Replace NaN with 0 (neutral for clustering distance)
df_encoded = df_encoded.fillna(0)

print("Total NaN values after cleaning:", df_encoded.isna().sum().sum())

scaled_features = scaler.fit_transform(df_encoded)

df_scaled = pd.DataFrame(
    scaled_features,
    columns=df_encoded.columns
)

print("Preprocessing complete")

# -----------------------------
# 5. Save processed data
# -----------------------------
df_scaled.to_csv("outputs/processed_features.csv", index=False)

print("Saved processed_features.csv to outputs/")

# Save scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save feature names
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(df_encoded.columns.tolist(), f)

print("Scaler and feature names saved")
