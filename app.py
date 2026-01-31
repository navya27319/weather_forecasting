import streamlit as st
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 🧠 TRAIN A SIMPLE MODEL
# -----------------------------
# For demonstration, we’ll train a small dummy model inside the same script.
# In real apps, you would train separately and load with pickle.

def train_model():
    # Example data (Team1, Team2, Venue, Weather, Importance)
    X = np.array([
        [0, 1, 0, 0, 5],
        [1, 0, 1, 1, 7],
        [2, 3, 0, 2, 3],
        [3, 2, 2, 0, 9],
        [0, 2, 1, 1, 4],
        [1, 3, 2, 2, 6]
    ])
    y = np.array(["Team A", "Team A", "Team C", "Team D", "Team A", "Team B"])

    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# Try loading a pre-trained model if available, otherwise train a new one
try:
    model = pickle.load(open("model.pkl", "rb"))
except:
    model = train_model()
    pickle.dump(model, open("model.pkl", "wb"))

# -----------------------------
# 🎨 FRONTEND UI WITH STREAMLIT
# -----------------------------

st.set_page_config(page_title="Game Prediction App", page_icon="🏆", layout="centered")

st.title("🏆 Game Prediction App")
st.markdown("### Predict which team is most likely to win the match!")

# Custom styling
st.markdown("""
<style>
    .stApp {
        background-color: #f9fafc;
    }
    h1, h3 {
        text-align: center;
        color: #004aad;
    }
</style>
""", unsafe_allow_html=True)

# Input form
st.subheader("Enter Match Details")

teams = ["Team A", "Team B", "Team C", "Team D"]
team1 = st.selectbox("Select Team 1", teams, index=0)
team2 = st.selectbox("Select Team 2", [t for t in teams if t != team1])
venue = st.selectbox("Match Venue", ["Home", "Away", "Neutral"])
weather = st.selectbox("Weather Condition", ["Sunny", "Rainy", "Cloudy"])
importance = st.slider("Match Importance (1 = Friendly, 10 = Final)", 1, 10, 5)

# Convert inputs to numeric for model
def encode_features(t1, t2, v, w, imp):
    team_index = {"Team A": 0, "Team B": 1, "Team C": 2, "Team D": 3}
    venue_index = {"Home": 0, "Away": 1, "Neutral": 2}
    weather_index = {"Sunny": 0, "Rainy": 1, "Cloudy": 2}
    return np.array([[team_index[t1], team_index[t2], venue_index[v], weather_index[w], imp]])

# Predict button
if st.button("⚡ Predict Winner"):
    input_data = encode_features(team1, team2, venue, weather, importance)
    prediction = model.predict(input_data)[0]

    st.success(f"🏁 *Predicted Winner:* {prediction}")
    st.balloons()

st.markdown("---")
st.caption("⚙ Built with Streamlit & Random Forest | © 2025 Game Predictor")