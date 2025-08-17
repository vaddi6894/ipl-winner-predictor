# 🏏 IPL Winner Prediction

This project predicts the winner of an **IPL (Indian Premier League)** cricket match using **Machine Learning**.  
It includes a **Streamlit web app** where users can select teams, toss details, and match conditions to get a real-time prediction.

---

## ✨ Features of the Website
- 🖥️ **Interactive Web App** built with Streamlit  
- 🔮 **Predict Match Winner** by selecting:
  - Team 1 & Team 2  
  - Toss winner  
  - Toss decision (bat/field)  
  - Match result type (normal/tie/no result)  
- 📊 **Visualizations included during training**:
  - Team wins distribution  
  - Toss decision stats  
  - Heatmap of feature correlations  
- ✅ Handles special cases (like *tie* or *no result*) gracefully  
- 📂 Easy deployment on **Streamlit Cloud**  


---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/IPL-Winner-Prediction.git
   cd IPL-Winner-Prediction

python -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate      # On Windows

pip install -r requirements.txt
python train_ipl_model.py --data_path matches.csv --artifacts_dir artifacts
streamlit run app.py

📊 Dataset

Source: IPL historical matches dataset (matches.csv)

Preprocessing includes:

Handling missing values

Updating old franchise names (Delhi Daredevils → Delhi Capitals, etc.)

Encoding categorical variables

📈 Model Training

Algorithm: Random Forest Classifier

Steps:

Feature Engineering (teams, toss, result type, etc.)

One-Hot Encoding & Label Encoding

Train/Test split

Scaling with StandardScaler

Model saved using Joblib (artifacts/model.pkl)

🚀 Deployment

This app can be deployed easily on Streamlit Cloud:

streamlit run app.py




