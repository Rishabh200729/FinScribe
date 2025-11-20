import streamlit as st
import requests

FASTAPI_URL = "http://localhost:8000"

st.title("FinScribe – Transaction Categorisation AI")
st.write("Zero-Shot + Self-Improving Financial Categoriser")

transaction = st.text_input("Enter Transaction Description:")

if st.button("Predict Category"):
    if transaction.strip() == "":
        st.warning("Please enter a transaction.")
    else:
        response = requests.post(f"{FASTAPI_URL}/predict", json={"transaction": transaction})
        result = response.json()
        print(result)
        st.subheader("Prediction")
        st.write(f"**Category:** {result}")
        # st.write(f"**Confidence:** {result['confidence']}")

        st.subheader("Top 3 Predictions")
        # for c in result["top_3"]:
        #     st.write(f"{c['category']} — {round(c['score'], 4)}")

        # # Feedback UI
        # st.subheader("Was the category wrong?")
        # correct_cat = st.text_input("Enter Correct Category:")

        # if st.button("Submit Feedback"):
        #     if correct_cat.strip() != "":
        #         fb = requests.post(f"{FASTAPI_URL}/feedback", json={
        #             "transaction": transaction,
        #             "correct_category": correct_cat
        #         })
        #         st.success("Feedback submitted! The model is now better.")
        #     else:
        #         st.warning("Please enter a valid category.")
