# app.py

import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image

# Load artifacts
model = joblib.load("artifacts/ipl_rf_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
label_encoder = joblib.load("artifacts/label_encoder.pkl")
train_features = joblib.load("artifacts/train_features.pkl")
metrics = joblib.load("artifacts/metrics.pkl")

# Streamlit page config
st.set_page_config(page_title="🏏 IPL Match Winner Predictor", layout="centered")

# Sidebar - Model Performance
st.sidebar.title("📊 Model Performance")

# Show metrics
st.sidebar.metric("✅ Accuracy", f"{metrics['accuracy']*100:.2f}%")
st.sidebar.metric("✅ F1-score", f"{metrics['f1_score']*100:.2f}%")

# Show confusion matrix
cm_path = "artifacts/confusion_matrix.png"
if os.path.exists(cm_path):
    st.sidebar.image(Image.open(cm_path), caption="Confusion Matrix")
else:
    st.sidebar.warning("⚠️ Confusion matrix not found. Run training script first.")

# Main app
st.title("🏏 IPL Match Winner Predictor")
st.markdown("### Predict the winner of an IPL match based on team and toss details")

# Teams list
teams = list(label_encoder.classes_)
toss_decisions = ["bat", "field"]
result_types = ["normal", "tie", "no result"]

# User input
team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", [t for t in teams if t != team1])
toss_winner = st.selectbox("Select Toss Winner", [team1, team2])
toss_decision = st.selectbox("Select Toss Decision", toss_decisions)
match_result = st.selectbox("Select Match Result Type", result_types)

# Predict button
if st.button("🔮 Predict Winner"):
    if match_result == "no result":
        st.warning("⚠️ Match has no result, so no winner can be predicted.")
    else:
        # Build input dataframe
        input_data = {
            "team1": [team1],
            "team2": [team2],
            "toss_winner": [toss_winner],
            "toss_decision": [toss_decision],
            "result": [match_result],
        }

        input_df = pd.DataFrame(input_data)

        # One-hot encode
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=train_features, fill_value=0)

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        winner = label_encoder.inverse_transform([prediction])[0]

        st.success(f"🏆 Predicted Winner: **{winner}**")
