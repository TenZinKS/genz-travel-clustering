from flask import Flask, render_template, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd
import numpy as np
import pickle
import sqlite3
from pathlib import Path

app = Flask(__name__)
app.secret_key = "dev_secret_change_me"

DB_PATH = Path(__file__).resolve().parent / "users.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                cluster INTEGER NOT NULL,
                recommendation TEXT NOT NULL,
                explanation TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )
        conn.commit()

model = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")
with open("models/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

RECOMMENDATIONS = {
    0: "Budget-friendly, trend-forward city breaks shaped by popular, fast-moving travel trends.",
    1: "Slow, scenic trips focused on relaxation and low reliance on social influence.",
    2: "Experience-heavy adventure travel with curated, photo-worthy moments.",
    3: "Balanced plans combining popular spots with personal discovery and comfort.",
}

EXPLANATIONS = {
    0: "You lean toward fast, affordable, and trend-driven trips. That usually means short city breaks, viral spots, and flexible plans that are easy to share.",
    1: "You show a calmer content style and lower influence from social media. That points to slower travel, fewer hotspots, and more time for rest and reflection.",
    2: "You engage deeply with travel content and visuals. That often correlates with experiential, story-rich trips where every day has a highlight.",
    3: "You balance inspiration from social media with personal preferences. That suggests a mix of popular sights and low-key discoveries.",
}

init_db()

@app.route("/")
@app.route("/login")
@app.route("/register")
@app.route("/dashboard")
def react_app():
    return render_template("app.html")

@app.route("/api/me")
def api_me():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"authenticated": False}), 200
    with get_db() as conn:
        row = conn.execute("SELECT id, name, email FROM users WHERE id = ?", (user_id,)).fetchone()
        rec = conn.execute(
            """
            SELECT cluster, recommendation, explanation, created_at
            FROM recommendations
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
    if not row:
        return jsonify({"authenticated": False}), 200
    last_recommendation = None
    if rec:
        last_recommendation = {
            "cluster": rec["cluster"],
            "recommendation": rec["recommendation"],
            "explanation": rec["explanation"],
            "created_at": rec["created_at"],
        }
    return jsonify(
        {
            "authenticated": True,
            "user": {
                "id": row["id"],
                "name": row["name"],
                "email": row["email"],
                "last_recommendation": last_recommendation,
            },
        }
    )

@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not name or not email or not password:
        return jsonify({"ok": False, "message": "All fields are required."}), 400

    with get_db() as conn:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            return jsonify({"ok": False, "message": "Email already registered."}), 400
        password_hash = generate_password_hash(password)
        cursor = conn.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name, email, password_hash),
        )
        conn.commit()
        session["user_id"] = cursor.lastrowid
    return jsonify({"ok": True})

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(force=True, silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    with get_db() as conn:
        row = conn.execute(
            "SELECT id, password_hash FROM users WHERE email = ?",
            (email,),
        ).fetchone()
    if not row or not check_password_hash(row["password_hash"], password):
        return jsonify({"ok": False, "message": "Invalid email or password."}), 400

    session["user_id"] = row["id"]
    return jsonify({"ok": True})

@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.pop("user_id", None)
    return jsonify({"ok": True})

@app.route("/recommendation", methods=["GET", "POST"])
def recommendation():
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
        decision = ""
        travel_interest = request.form.get("self_reported_travel_interest", "")
        engagement_level = request.form.get("engages_with_travel_posts", "")

        user_features["daily_usage_hours"] = daily_usage_map.get(daily_hours, 0)
        user_features["usage_frequency"] = likert_map.get(usage_freq, 0)
        user_features["trust_in_social_media"] = likert_map.get(trust, 0)
        user_features["influencer_influence"] = influencer_map.get(influencer, 0)
        # Derive decision influence from trust + influencer
        trust_score = likert_map.get(trust, 0)
        influencer_score = influencer_map.get(influencer, 0)
        if trust_score == 3 or influencer_score == 3:
            decision = "High"
        elif trust_score == 2 or influencer_score == 2:
            decision = "Medium"
        else:
            decision = "Low"
        user_features["decision_influence_level"] = likert_map.get(decision, 0)
        user_features["self_reported_travel_interest"] = likert_map.get(travel_interest, 0)
        user_features["engages_with_travel_posts"] = engagement_map.get(engagement_level, 0)
        # Defaults derived to reduce repetitive questions
        user_features["feels_addicted"] = 1 if daily_hours in {"3-5", "5+"} else 0
        user_features["follows_travel_content"] = 1 if engagement_level == "Often" else 0
        user_features["saves_travel_content"] = 1 if engagement_level == "Often" else 0
        user_features["plans_trips_via_social_media"] = 1 if trust_score >= 2 or influencer_score >= 2 else 0

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
        explanation = EXPLANATIONS.get(
            cluster,
            "This profile suggests a balanced mix of inspiration and personal preferences."
        )

        user_id = session.get("user_id")
        if user_id:
            with get_db() as conn:
                conn.execute(
                    """
                    INSERT INTO recommendations (user_id, cluster, recommendation, explanation)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, cluster, recommendation, explanation),
                )
                conn.commit()

        return render_template(
            "result.html",
            cluster=cluster,
            recommendation=recommendation,
            explanation=explanation
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
