import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Swiggy Delivery Time Predictor", layout="centered")

# -----------------------------------------------------------------------
# Load the trained pipeline (preprocessing + tuned Random Forest model)
# -----------------------------------------------------------------------
PIPELINE_PATH = "swiggy_pipeline.pkl"


@st.cache_resource
def load_pipeline(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


try:
    pipeline = load_pipeline(PIPELINE_PATH)
    load_error = None
except FileNotFoundError:
    pipeline = None
    load_error = (
        f"Couldn't find '{PIPELINE_PATH}'. Place the pickle file created by the "
        f"notebook (`swiggy_pipeline.pkl`) in the same folder as this app.py."
    )

st.title("Swiggy Delivery Time Predictor")
st.write("Enter the order details below to estimate the delivery time (in minutes).")

if load_error:
    st.error(load_error)

# -----------------------------------------------------------------------
# Input form
# -----------------------------------------------------------------------
with st.form("order_form"):
    st.subheader("Rider details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Rider age", min_value=18, max_value=60, value=30)
    with col2:
        ratings = st.slider("Rider rating", min_value=1.0, max_value=5.0, value=4.6, step=0.1)

    vehicle_condition = st.select_slider(
        "Vehicle condition (0 = worst, 3 = best)", options=[0, 1, 2, 3], value=2
    )
    type_of_vehicle = st.selectbox(
        "Vehicle type", ["motorcycle", "scooter", "electric_scooter", "bicycle"]
    )
    multiple_deliveries = st.selectbox("Multiple deliveries on this trip", [0, 1, 2, 3], index=0)

    st.subheader("Order details")
    type_of_order = st.selectbox("Order type", ["snack", "meal", "drinks", "buffet"])
    festival = st.selectbox("Is there a festival?", ["no", "yes"])

    st.subheader("Location")
    col3, col4 = st.columns(2)
    with col3:
        restaurant_latitude = st.number_input("Restaurant latitude", value=12.9716, format="%.6f")
        restaurant_longitude = st.number_input("Restaurant longitude", value=77.5946, format="%.6f")
    with col4:
        delivery_latitude = st.number_input("Delivery latitude", value=12.9916, format="%.6f")
        delivery_longitude = st.number_input("Delivery longitude", value=77.6146, format="%.6f")

    city_name = st.text_input("City name / code (e.g. BANG, HYD, CHEN)", value="HYD")
    city_type = st.selectbox("City type", ["urban", "metropolitian", "semi-urban"])

    def haversine_km(lat1, lon1, lat2, lon2):
        r = 6371.0
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
        return 2 * r * np.arcsin(np.sqrt(a))

    auto_distance = haversine_km(
        restaurant_latitude, restaurant_longitude, delivery_latitude, delivery_longitude
    )
    distance = st.number_input(
        "Distance (km)", min_value=0.0, value=float(round(auto_distance, 2)), step=0.1,
        help="Auto-calculated from the coordinates above; adjust if you have the actual value.",
    )

    st.subheader("Timing & conditions")
    weather = st.selectbox("Weather", ["sunny", "cloudy", "stormy", "sandstorms", "fog", "windy"])
    traffic = st.selectbox("Traffic", ["low", "medium", "high", "jam"])

    order_date = st.date_input("Order date", value=datetime.today())
    order_day_of_week = order_date.strftime("%A").lower()
    is_weekend = 1 if order_date.weekday() >= 5 else 0
    st.caption(f"Day of week: **{order_day_of_week}** · Weekend: **{'Yes' if is_weekend else 'No'}**")

    col5, col6 = st.columns(2)
    with col5:
        order_time_hour = st.slider("Order hour (24h)", min_value=0, max_value=23, value=19)
    with col6:
        pickup_time_minutes = st.number_input(
            "Pickup time (minutes)", min_value=0.0, max_value=60.0, value=15.0, step=1.0
        )

    def time_of_day(hour):
        if hour < 6:
            return "after_midnight"
        if hour < 12:
            return "morning"
        if hour < 17:
            return "afternoon"
        if hour < 21:
            return "evening"
        return "night"

    order_time_of_day = time_of_day(order_time_hour)
    st.caption(f"Time of day bucket: **{order_time_of_day}**")

    submitted = st.form_submit_button("Predict delivery time")

# -----------------------------------------------------------------------
# Build the feature row exactly as the training pipeline expects
# (same columns as `x` in the notebook: df minus time_taken, rider_id, order_date)
# -----------------------------------------------------------------------
if submitted:
    if pipeline is None:
        st.error("Model pipeline not loaded — see the message above.")
    else:
        input_df = pd.DataFrame([{
            "age": age,
            "ratings": ratings,
            "restaurant_latitude": restaurant_latitude,
            "restaurant_longitude": restaurant_longitude,
            "delivery_latitude": delivery_latitude,
            "delivery_longitude": delivery_longitude,
            "weather": weather,
            "traffic": traffic,
            "vehicle_condition": vehicle_condition,
            "type_of_order": type_of_order,
            "type_of_vehicle": type_of_vehicle,
            "multiple_deliveries": multiple_deliveries,
            "festival": festival,
            "city_type": city_type,
            "city_name": city_name,
            "order_day": order_date.day,
            "order_month": order_date.month,
            "order_day_of_week": order_day_of_week,
            "is_weekend": is_weekend,
            "pickup_time_minutes": pickup_time_minutes,
            "order_time_hour": float(order_time_hour),
            "order_time_of_day": order_time_of_day,
            "distance": distance,
        }])

        prediction = pipeline.predict(input_df)[0]
        st.success(f"### Estimated delivery time: **{prediction:.1f} minutes**")

        with st.expander("Show input data sent to the model"):
            st.dataframe(input_df.T.rename(columns={0: "value"}))
