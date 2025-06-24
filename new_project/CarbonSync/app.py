import streamlit as st
from cp import ChainedPredictor

# Load model
model_paths = {
    'assd':     'models/assd_lstm_model.keras',
    'temp':     'models/temp.pkl',
    'sp':       'models/SP.pkl',
    'wind':     'models/wind_speed.pkl',
    'encoder1': 'models/encoder1.pkl',
    'scaler1':  'models/scaler1.pkl',
    'encoder2': 'models/encoder2.pkl',
    'scaler2':  'models/scaler2.pkl',
    'encoder3': 'models/encoder3.pkl',
    'scaler3':  'models/scaler3.pkl',
    'encoder4': 'models/encoder4.pkl',
    'scaler4':  'models/scaler4.pkl'
}
predictor = ChainedPredictor(model_paths)

# UI
st.title("🌿 CarbonSync Regional Forecast")
region = st.selectbox("Select Region", ["Asia"])
country = st.selectbox("Select Country", ["India"])
year = st.number_input("Year", min_value=2020, max_value=2030, value=2025)
month = st.slider("Month", 1, 12, 6)

if st.button("Predict"):
    result = predictor.predict(region, country, year, month)
    st.success("Prediction Complete!")
    st.json(result)
