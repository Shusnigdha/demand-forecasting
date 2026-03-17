import streamlit as st
import pandas as pd
import numpy as np
import pickle

@st.cache_resource
def load_artifacts():
    with open("demand_forecasting_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    return model, encoders

model, label_encoders = load_artifacts()

st.title("Demand Forecasting App")
st.divider()
st.header("Input Features :")

epidemic_input = st.selectbox("Epidemic", ["No", "Yes"])
epidemic = 1 if epidemic_input == "Yes" else 0

promotion_input = st.selectbox("Promotion", ["No", "Yes"])
promotion = 1 if promotion_input == "Yes" else 0

category = st.selectbox("Category", label_encoders["Category"].classes_.tolist())
price = st.number_input("Price", min_value= 0.0, value= 50.0)
weather_condition = st.selectbox("Weather Condition", label_encoders["Weather Condition"].classes_.tolist())
region = st.selectbox("Region", label_encoders["Region"].classes_.tolist())
discount = st.number_input("Discount (%)", min_value = 0, max_value= 100, value= 10)
seasonality = st.selectbox("Season", label_encoders["Seasonality"].classes_.tolist())
competitor_pricing = st.number_input("Competitor Price", min_value= 0.0, value= 50.0)
inventory_level = st.number_input("Inventory Level", min_value= 0, value= 100)

months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
selected_month = st.selectbox("Month", months)
month = months.index(selected_month) + 1

week_day = st.selectbox("WeekDay", label_encoders["WeekDay"].classes_.tolist())

input_data = pd.DataFrame({
    "Epidemic" : [epidemic],
    "Promotion" : [promotion],
    "Category" : [category],
    "Price" : [price],
    "Weather Condition" : [weather_condition],
    "Region" : [region],
    "Discount" : [discount],
    "Seasonality" : [seasonality],
    "Competitor Pricing" : [competitor_pricing],
    "Inventory Level" : [inventory_level],
    "Month" : [month],
    "WeekDay" : [week_day]
})

for col, encoder in label_encoders.items():
    if col in input_data.columns:
        input_data[col] = encoder.transform(input_data[col])

st.divider()

if st.button("Predict Demand"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Demand: {int(prediction)}Units")