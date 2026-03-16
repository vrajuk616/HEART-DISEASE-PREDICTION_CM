import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp {
background: linear-gradient(135deg,#1f4037,#99f2c8);
}

h1 {
color:white;
text-align:center;
}

.block-container {
padding-top:2rem;
}

.prediction-box {
background-color:white;
padding:20px;
border-radius:15px;
text-align:center;
font-size:22px;
font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("heart_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- HEADER ----------------
st.title("❤️ Heart Disease Prediction Dashboard")

st.image(
"https://images.unsplash.com/photo-1588776814546-ec7e4c7f7d5d",
use_container_width=True
)

st.write(
"AI powered system to predict the probability of heart disease using Machine Learning."
)

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("Patient Details")

age = st.sidebar.slider("Age", 20, 80, 40)
sex = st.sidebar.selectbox("Sex", [0,1])
cp = st.sidebar.slider("Chest Pain Type", 0,3,1)
trestbps = st.sidebar.slider("Resting Blood Pressure", 90,200,120)
chol = st.sidebar.slider("Cholesterol", 100,600,200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar", [0,1])
restecg = st.sidebar.slider("Rest ECG",0,2,1)
thalach = st.sidebar.slider("Max Heart Rate",60,220,150)
exang = st.sidebar.selectbox("Exercise Induced Angina",[0,1])
oldpeak = st.sidebar.slider("Oldpeak",0.0,6.0,1.0)
slope = st.sidebar.slider("Slope",0,2,1)
ca = st.sidebar.slider("Major Vessels",0,4,0)
thal = st.sidebar.slider("Thal",0,3,2)

# ---------------- PREDICTION ----------------
input_data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,
                        thalach,exang,oldpeak,slope,ca,thal]])

input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)
probability = model.predict_proba(input_scaled)[0][1]

col1,col2 = st.columns(2)

# ---------------- RESULT ----------------
with col1:

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.metric("Risk Probability", f"{probability*100:.2f}%")

# ---------------- PROBABILITY CHART ----------------
with col2:

    st.subheader("Prediction Probability")

    fig, ax = plt.subplots()

    labels = ["No Disease","Disease"]
    values = [1-probability, probability]

    ax.bar(labels, values)

    ax.set_ylabel("Probability")

    st.pyplot(fig)

# ---------------- FEATURE IMPORTANCE ----------------
st.subheader("Model Feature Importance")

features = [
"Age","Sex","Chest Pain","BP","Cholesterol","FBS","ECG",
"Heart Rate","Exercise Angina","Oldpeak","Slope","CA","Thal"
]

importances = model.feature_importances_

importance_df = pd.DataFrame({
"Feature":features,
"Importance":importances
})

importance_df = importance_df.sort_values("Importance",ascending=False)

fig2, ax2 = plt.subplots(figsize=(8,5))

sns.barplot(
x="Importance",
y="Feature",
data=importance_df,
ax=ax2
)

st.pyplot(fig2)

# ---------------- FOOTER ----------------
st.markdown("---")

st.write("Machine Learning Model: Random Forest")

st.write("Built with Streamlit")