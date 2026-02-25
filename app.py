# ==========================================================
# 🧬 Novagen Research Lab
# Health Risk Prediction System
# Random Forest Classifier (93% Accuracy)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

# ----------------------------------------------------------
# 🔹 Page Configuration
# ----------------------------------------------------------
st.set_page_config(
    page_title="Novagen Health Risk Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# 🔹 Custom Styling
# ----------------------------------------------------------
st.markdown("""
<style>
.main {background-color: #f4f6f9;}
.stButton>button {
    background-color: #0066cc;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# 🔹 Load Model Safely
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = Path("model.pkl")
    if not model_path.exists():
        st.error("⚠️ model.pkl not found in project folder.")
        return None
    
    model = joblib.load(model_path)
    return model

model = load_model()

# ----------------------------------------------------------
# 🔹 Header
# ----------------------------------------------------------
st.title("🧬 Novagen Research Lab")
st.subheader("AI-Based Health Risk Prediction System")
st.markdown("---")

st.write("""
This AI system predicts whether an individual is:

- ✅ **Healthy (0)**
- ⚠️ **Unhealthy (1)**  

Built using a **Random Forest Classifier** with **93% accuracy**.
""")

# ----------------------------------------------------------
# 🔹 Sidebar Inputs
# ----------------------------------------------------------
st.sidebar.header("📝 Patient Parameters")

def get_user_input():

    Age = st.sidebar.slider("Age", 10, 100, 30)
    BMI = st.sidebar.slider("BMI", 10.0, 45.0, 22.5)
    Blood_Pressure = st.sidebar.slider("Blood Pressure", 80, 200, 120)
    Cholesterol = st.sidebar.slider("Cholesterol", 100, 350, 180)
    Glucose_Level = st.sidebar.slider("Glucose Level", 70, 250, 100)
    Heart_Rate = st.sidebar.slider("Heart Rate", 50, 150, 72)
    Sleep_Hours = st.sidebar.slider("Sleep Hours", 0.0, 12.0, 7.0)
    Exercise_Hours = st.sidebar.slider("Exercise Hours/Week", 0.0, 20.0, 3.0)
    Water_Intake = st.sidebar.slider("Water Intake (Liters/Day)", 0.5, 5.0, 2.0)
    Stress_Level = st.sidebar.slider("Stress Level (1-10)", 1, 10, 5)

    Smoking = st.sidebar.selectbox("Smoking", [0, 1])
    Alcohol = st.sidebar.selectbox("Alcohol", [0, 1])
    Diet = st.sidebar.selectbox("Balanced Diet", [0, 1])
    MentalHealth = st.sidebar.selectbox("Mental Health Issues", [0, 1])
    PhysicalActivity = st.sidebar.selectbox("Physically Active", [0, 1])
    MedicalHistory = st.sidebar.selectbox("Medical History", [0, 1])
    Allergies = st.sidebar.selectbox("Allergies", [0, 1])

    Diet_Type = st.sidebar.selectbox("Diet Type", ["Non-Vegetarian", "Vegan", "Vegetarian"])
    Blood_Group = st.sidebar.selectbox("Blood Group", ["A", "AB", "B", "O"])

    # Manual One-Hot Encoding
    Diet_Type__Vegan = 1 if Diet_Type == "Vegan" else 0
    Diet_Type__Vegetarian = 1 if Diet_Type == "Vegetarian" else 0

    Blood_Group_AB = 1 if Blood_Group == "AB" else 0
    Blood_Group_B = 1 if Blood_Group == "B" else 0
    Blood_Group_O = 1 if Blood_Group == "O" else 0

    data = {
        'Age': Age,
        'BMI': BMI,
        'Blood_Pressure': Blood_Pressure,
        'Cholesterol': Cholesterol,
        'Glucose_Level': Glucose_Level,
        'Heart_Rate': Heart_Rate,
        'Sleep_Hours': Sleep_Hours,
        'Exercise_Hours': Exercise_Hours,
        'Water_Intake': Water_Intake,
        'Stress_Level': Stress_Level,
        'Smoking': Smoking,
        'Alcohol': Alcohol,
        'Diet': Diet,
        'MentalHealth': MentalHealth,
        'PhysicalActivity': PhysicalActivity,
        'MedicalHistory': MedicalHistory,
        'Allergies': Allergies,
        'Diet_Type__Vegan': Diet_Type__Vegan,
        'Diet_Type__Vegetarian': Diet_Type__Vegetarian,
        'Blood_Group_AB': Blood_Group_AB,
        'Blood_Group_B': Blood_Group_B,
        'Blood_Group_O': Blood_Group_O
    }

    return pd.DataFrame([data])

input_df = get_user_input()

# ----------------------------------------------------------
# 🔹 Align Columns with Model
# ----------------------------------------------------------
if model is not None and hasattr(model, "feature_names_in_"):
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# ----------------------------------------------------------
# 🔹 Prediction Section (Stable Version - No NameError)
# ----------------------------------------------------------
st.markdown("## 🔍 Prediction Result")

# Initialize session state
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.probabilities = None

if st.button("🚀 Predict Health Status"):

    if model is not None:

        st.session_state.prediction = model.predict(input_df)[0]
        st.session_state.probabilities = model.predict_proba(input_df)[0]

# ----------------------------------------------------------
# 🔹 Display Result (Only If Available)
# ----------------------------------------------------------
if st.session_state.prediction is not None:

    prediction = st.session_state.prediction
    probabilities = st.session_state.probabilities

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("### 🧾 Health Assessment")

        if str(prediction).lower() == "healthy":
            st.success(f"✅ Status: **{prediction}**")
        else:
            st.error(f"⚠️ Status: **{prediction}**")

    with col2:
        st.metric(label="Model Accuracy", value="93%")

    # ------------------------------------------------------
    # 🔹 Confidence Visualization
    # ------------------------------------------------------
    st.markdown("### 📊 Prediction Confidence Level")

    class_labels = model.classes_

    prob_df = pd.DataFrame({
        "Health Status": class_labels,
        "Confidence Score": probabilities
    })

    fig = px.bar(
        prob_df,
        x="Health Status",
        y="Confidence Score",
        text_auto=".2f",
        color="Health Status"
    )

    fig.update_layout(
        yaxis_title="Probability",
        xaxis_title="",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------
    # 🔹 Clinical Insight
    # ------------------------------------------------------
    st.markdown("### 🧠 Clinical Insight")

    if str(prediction).lower() == "healthy":
        st.info(
            "The individual falls within the healthy risk range based on the provided parameters. "
            "Maintain current lifestyle habits and continue preventive monitoring."
        )
    else:
        st.warning(
            "The individual shows elevated health risk indicators. "
            "Lifestyle adjustments and medical consultation are recommended."
        )

        
# ----------------------------------------------------------
# 🔹 Feature Importance (Safe Version)
# ----------------------------------------------------------
if model is not None and hasattr(model, "feature_importances_"):

    st.markdown("---")
    st.markdown("## 📈 Feature Importance Analysis")

    importance_df = pd.DataFrame({
        "Feature": model.feature_names_in_,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig_importance = px.bar(
        importance_df.head(10),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 10 Most Influential Features"
    )

    st.plotly_chart(fig_importance, use_container_width=True)

# ----------------------------------------------------------
# 🔹 Footer
# ----------------------------------------------------------
st.markdown("---")
st.markdown("© 2026 Novagen Research Lab | AI for Preventive Healthcare")