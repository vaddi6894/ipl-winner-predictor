# train_ipl_model.py

import pandas as pd
import numpy as np
import argparse
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def main(data_path, artifacts_dir):
    # Load dataset
    data = pd.read_csv(data_path)

    # Drop unnecessary columns
    data = data.iloc[:, :-1]  # drop last column if unnecessary
    data.dropna(inplace=True)
    data.drop(
        ["id", "Season", "city", "date", "player_of_match", "venue", "umpire1", "umpire2"],
        axis=1,
        inplace=True,
    )

    # Handle inconsistent team names
    replacements = {
        "Delhi Daredevils": "Delhi Capitals",
        "Deccan Chargers": "Sunrisers Hyderabad",
        "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
        "Kings XI Punjab": "Punjab Kings",
    }
    for old, new in replacements.items():
        data.replace(old, new, inplace=True)

    # Remove matches with "no result" (no winner possible)
    data = data[data["result"] != "no result"]

    # Features & target
    X = data.drop(["winner"], axis=1)
    y = data["winner"]

    # One-hot encode categorical features
    X = pd.get_dummies(
        X, columns=["team1", "team2", "toss_winner", "toss_decision", "result"], drop_first=True
    )

    # Label encode target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predictions & evaluation
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)

    print("✅ Training complete.")
    print(f"📊 Accuracy: {acc:.4f}")
    print(f"📊 F1-score: {f1:.4f}")

    # Save artifacts
    os.makedirs(artifacts_dir, exist_ok=True)
    joblib.dump(model, f"{artifacts_dir}/ipl_rf_model.pkl")
    joblib.dump(scaler, f"{artifacts_dir}/scaler.pkl")
    joblib.dump(label_encoder, f"{artifacts_dir}/label_encoder.pkl")
    joblib.dump(X.columns.tolist(), f"{artifacts_dir}/train_features.pkl")

    # Save confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        fmt="d",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.title("Confusion Matrix - IPL Winner Prediction")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{artifacts_dir}/confusion_matrix.png")
    plt.close()

    # Save metrics
    metrics = {"accuracy": acc, "f1_score": f1}
    joblib.dump(metrics, f"{artifacts_dir}/metrics.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to matches.csv dataset")
    parser.add_argument("--artifacts_dir", type=str, required=True, help="Directory to save artifacts")
    args = parser.parse_args()

    main(args.data_path, args.artifacts_dir)
