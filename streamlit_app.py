import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
dv = pickle.load(open("dv.pkl", "rb"))

st.title("Customer Churn Prediction")

# Example inputs (adjust to your features)
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.slider("Tenure", 0, 72)
monthly_charges = st.number_input("Monthly Charges")

if st.button("Predict"):
    customer = {
        "gender": gender,
        "tenure": tenure,
        "monthlycharges": monthly_charges
    }

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0][1]

    st.write(f"Churn probability: {y_pred:.2f}")