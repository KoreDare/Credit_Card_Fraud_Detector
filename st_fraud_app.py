import streamlit as st
import numpy as np
import joblib
import pandas as pd

#Loading the trained model
model = joblib.load('fraud_xgb_model.pkl')

#Loading the scaler too
scaler = joblib.load('scaler.pkl')

st.title("💳AI-Powered Credit Card Fraud Detector")

st.markdown("Enter the transaction details below:")
st.subheader("📁 Or Upload Multiple Transactions")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file) #Read the file
        if 'Class' in input_df.columns:
            input_df = input_df.drop('Class', axis=1)

        #Show uploaded data
        st.write("Preview of uploaded data:")
        st.dataframe(input_df.head())

        input_df[['Time', 'Amount']] = scaler.transform(input_df[['Time', 'Amount']])

        predictions = model.predict(input_df)

        input_df['Prediction'] = predictions
        input_df['Prediction_Label'] = input_df['Prediction'].apply(lambda x: "Fraud" if x == 1 else "Not Fraud")

        st.subheader("🧾 Prediction Results(Top 100 rows):")

        def highlight_fraud(row):
            color = 'background-color: red; color: white' if row['Prediction_Label'] == 'Fraud' else 'background-color: green; color: white'
            return [color] * len(row)

        styled_df = input_df.head(100).style.apply(highlight_fraud, axis=1)
        st.dataframe(styled_df)


    except Exception as e:
        st.error(f"❌Error processing file:{e}")

#Inputting features: Time, V1-v28, Amount
st.markdown("---")
st.subheader("✍️ Or Enter a Single Transaction Manually")

time = st.number_input("Time", value=0.0)
amount = st.number_input("Amount", value=0.0)

v_features = []
for i in range(1, 29):
    v = st.number_input(f"V{i}", value=0.0)
    v_features.append(v)

#Scaling Time and Amount
scaled_time_amount = scaler.transform([[time, amount]])[0]
scaled_time = scaled_time_amount[0]
scaled_amount = scaled_time_amount[1]

#Combine input into a NumPy array for prediction
input_data = np.array([[time, *v_features, amount]])

if st.button("Predict Fraud"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️This transaction is likely FRAUDULENT!")
    else:
        st.success("✅This transaction is NOT fraudulent.")
 
