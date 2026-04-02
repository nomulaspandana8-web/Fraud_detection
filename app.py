import streamlit as st
from model import load_model
from utils import preprocess_input

# Load model
model, scaler, columns = load_model()

st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.markdown("<h1 style='text-align:center;'>💳 Fraud Detection System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Real-Time Transaction Analysis</h4>", unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Enter Transaction Details")

    time = st.number_input("Time", min_value=0.0)
    amount = st.number_input("Amount", min_value=0.0)

    v1 = st.number_input("V1")
    v2 = st.number_input("V2")
    v3 = st.number_input("V3")

    check = st.button("🚀 Check Transaction")

with col2:
    st.subheader("📊 Result")

    if check:
        input_data = {
            "Time": time,
            "Amount": amount,
            "V1": v1,
            "V2": v2,
            "V3": v3
        }

        processed = preprocess_input(input_data, scaler, columns)

        prediction = model.predict(processed)[0]
        prob = model.predict_proba(processed)[0][1]

        if prediction == 1:
            st.error("⚠️ Fraud Detected")
            st.metric("Fraud Probability", f"{prob:.2%}")
        else:
            st.success("✅ Legitimate Transaction")
            st.metric("Fraud Probability", f"{prob:.2%}")

    else:
        st.info("Enter details and click button")

st.markdown("---")

col3, col4, col5 = st.columns(3)

col3.metric("Model", "Random Forest")
col4.metric("Type", "ML Based")
col5.metric("Response", "Real-Time ⚡")