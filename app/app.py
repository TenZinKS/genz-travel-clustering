from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

model = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")
with open("models/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

RECOMMENDATIONS = {
    0: "Budget-friendly, trend-forward city breaks with heavy TikTok inspiration.",
    1: "Slow, scenic trips focused on relaxation and low social media involvement.",
    2: "Experience-heavy adventure travel with curated, Instagrammable moments.",
    3: "Balanced plans combining popular spots with personal discovery and comfort.",
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        # 1️⃣ Create empty feature dictionary
        user_features = {feature: 0 for feature in feature_names}

        # 2️⃣ Fill numeric features from form (match training bins)
        daily_usage_map = {
            "<1": 1,
            "1-2": 2,
            "3-5": 3,
            "5+": 4,
        }
        likert_map = {"Low": 1, "Medium": 2, "High": 3}
        influencer_map = {"No": 1, "Sometimes": 2, "Yes": 3}
        engagement_map = {"Rarely": 1, "Sometimes": 2, "Often": 3}

        daily_hours = request.form.get("daily_usage_hours", "")
        usage_freq = request.form.get("usage_frequency", "")
        trust = request.form.get("trust_in_social_media", "")
        influencer = request.form.get("influencer_influence", "")
        decision = request.form.get("decision_influence_level", "")
        travel_interest = request.form.get("self_reported_travel_interest", "")
        engagement_level = request.form.get("engages_with_travel_posts", "")

        user_features["daily_usage_hours"] = daily_usage_map.get(daily_hours, 0)
        user_features["usage_frequency"] = likert_map.get(usage_freq, 0)
        user_features["trust_in_social_media"] = likert_map.get(trust, 0)
        user_features["influencer_influence"] = influencer_map.get(influencer, 0)
        user_features["decision_influence_level"] = likert_map.get(decision, 0)
        user_features["self_reported_travel_interest"] = likert_map.get(travel_interest, 0)
        user_features["feels_addicted"] = int(request.form.get("feels_addicted", 0))
        user_features["follows_travel_content"] = int(request.form.get("follows_travel_content", 0))
        user_features["engages_with_travel_posts"] = engagement_map.get(engagement_level, 0)
        user_features["saves_travel_content"] = int(request.form.get("saves_travel_content", 0))
        user_features["plans_trips_via_social_media"] = int(request.form.get("plans_trips_via_social_media", 0))

        # 3️⃣ Handle one-hot encoded platform
        platform = request.form.get("primary_platform", "")
        if platform == "Instagram":
            user_features["primary_platform_Instagram"] = 1
        elif platform == "TikTok":
            user_features["primary_platform_TikTok"] = 1
        elif platform == "YouTube":
            user_features["primary_platform_YouTube"] = 1

        # 4️⃣ Handle content consumption style
        if request.form.get("content_consumption_style") == "Passive":
            user_features["content_consumption_style_Passive"] = 1

        # 5️⃣ Handle visual preference
        visual = request.form.get("visual_content_preference")
        if visual == "Low":
            user_features["visual_content_preference_Low"] = 1
        elif visual == "Medium":
            user_features["visual_content_preference_Medium"] = 1

        # 6️⃣ Convert to DataFrame and scale to match training
        X = pd.DataFrame([user_features], columns=feature_names)
        X_scaled = scaler.transform(X)

        # 7️⃣ Predict cluster
        cluster = int(model.predict(X_scaled)[0])

        recommendation = RECOMMENDATIONS.get(
            cluster,
            "A mixed travel style: explore popular highlights and add personal twists."
        )

        return render_template(
            "result.html",
            cluster=cluster,
            recommendation=recommendation
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
