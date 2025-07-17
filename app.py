import streamlit as st
import numpy as np
import pickle



with open(r'models\scaler.pkl','rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

with open(r'models\model.pkl','rb') as model_file:
    loaded_model = pickle.load(model_file)



st.title("E-commerce Predictor")

avg_session_length = st.number_input("Average Session Lenght")
time_on_app = st.number_input('Time on App')
length_of_membership = st.number_input("Length of Membership")

if st.button("Predict"):
    data = np.array([avg_session_length,time_on_app,length_of_membership]).reshape(1,-1)
    new_data = loaded_scaler.transform(data)
    Prediction = loaded_model.predict(new_data)

    st.success(f"Yearly Amount Spent is : $ {Prediction[0]}")

