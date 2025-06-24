
import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

# Load trained model and scaler
model = pickle.load(open("wine_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
accuracy = pickle.load(open("accuracy.pkl", "rb"))  # Load accuracy

# Streamlit App Title
st.title("🍷 Wine Quality Prediction App")

# Sidebar for User Input
st.sidebar.header("Enter Wine Features")
fixed_acidity = st.sidebar.number_input("Fixed Acidity", 4.0, 16.0, step=0.1)
volatile_acidity = st.sidebar.number_input("Volatile Acidity", 0.1, 2.0, step=0.01)
citric_acid = st.sidebar.number_input("Citric Acid", 0.0, 1.5, step=0.01)
residual_sugar = st.sidebar.number_input("Residual Sugar", 0.5, 15.0, step=0.1)
chlorides = st.sidebar.number_input("Chlorides", 0.01, 0.2, step=0.001)
free_sulfur_dioxide = st.sidebar.number_input("Free Sulfur Dioxide", 1.0, 75.0, step=1.0)
total_sulfur_dioxide = st.sidebar.number_input("Total Sulfur Dioxide", 5.0, 200.0, step=5.0)
density = st.sidebar.number_input("Density", 0.9900, 1.0050, step=0.0001)
pH = st.sidebar.number_input("pH", 2.5, 4.5, step=0.01)
sulphates = st.sidebar.number_input("Sulphates", 0.3, 1.5, step=0.01)
alcohol = st.sidebar.number_input("Alcohol", 8.0, 15.0, step=0.1)

# Button to Predict Quality
if st.sidebar.button("Predict Wine Quality"):
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                          free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)
    confidence = model.predict_proba(scaled_features)[0][1]  # Confidence score

    quality = "Good 🍷" if prediction[0] == 1 else "Bad 🍷"
    st.subheader(f"Predicted Wine Quality: **{quality}**")
    st.write(f"**Confidence Score:** {confidence:.2%}")

    # Feature Importance Plot
    importance = abs(model.coef_[0])
    feature_names = ['Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 'Residual Sugar', 
                     'Chlorides', 'Free Sulfur Dioxide', 'Total Sulfur Dioxide', 'Density', 
                     'pH', 'Sulphates', 'Alcohol']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feature_names, importance, color="skyblue")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance in Prediction")
    st.pyplot(fig)

    # Radar Chart for User Input
    fig_radar = px.line_polar(pd.DataFrame({"Feature": feature_names, "Value": scaled_features[0]}),
                              r="Value", theta="Feature", line_close=True)
    st.plotly_chart(fig_radar)

# Display Model Accuracy
st.sidebar.subheader(f"Model Accuracy: **{accuracy:.2f}**")
