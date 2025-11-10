# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Food Business Success Predictor", layout="centered")

# -------------------------
# Configuration (match training)
# -------------------------
MODEL_PATH = "models/food_success_model.pkl"
SCALER_PATH = "models/scaler.pkl"

feature_cols = [
    "footfall_index",
    "nearby_shops_count",
    "nearby_offices_count",
    "nearby_colleges_count",
    "nearby_hospitals_count",
    "nearby_restaurants_count",
    "nearby_parks_count",
    "distance_to_nearest_brand_chai",
    "rent_estimate",
    "avg_income_area"
]

# -------------------------
# Load model + scaler (with safety checks)
# -------------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")
    st.stop()

# -------------------------
# UI Inputs (unique keys)
# -------------------------
st.title("🍴 Food Business Success Predictor")
st.write("Enter site features — app scales & orders inputs exactly as the model expects.")

col1, col2 = st.columns(2)
with col1:
    footfall_index = st.number_input("Footfall index (people/day)", min_value=0, max_value=50000, value=50, step=1, key="ft")
    nearby_shops_count = st.number_input("Nearby shops count", min_value=0, max_value=500, value=2, step=1, key="shops")
    nearby_offices_count = st.number_input("Nearby offices count", min_value=0, max_value=500, value=1, step=1, key="offices")
    nearby_colleges_count = st.number_input("Nearby colleges count", min_value=0, max_value=50, value=0, step=1, key="colleges")
    nearby_hospitals_count = st.number_input("Nearby hospitals count", min_value=0, max_value=50, value=0, step=1, key="hospitals")

with col2:
    nearby_restaurants_count = st.number_input("Nearby restaurants count", min_value=0, max_value=500, value=3, step=1, key="restaurants")
    nearby_parks_count = st.number_input("Nearby parks count", min_value=0, max_value=200, value=0, step=1, key="parks")
    distance_to_nearest_brand_chai = st.number_input("Distance to nearest brand outlet (meters)", min_value=0, max_value=10000, value=200, step=1, key="dist")
    rent_estimate = st.number_input("Rent estimate (₹ per sq ft)", min_value=0.0, max_value=10000.0, value=50.0, step=1.0, key="rent")
    avg_income_area = st.number_input("Avg income of area (₹k / month)", min_value=0.0, max_value=500.0, value=20.0, step=1.0, key="income")

debug = st.checkbox("Show debug inputs (for troubleshooting)", value=False)

# -------------------------
# Build input DataFrame in exact order
# -------------------------
input_dict = {
    "footfall_index": float(footfall_index),
    "nearby_shops_count": float(nearby_shops_count),
    "nearby_offices_count": float(nearby_offices_count),
    "nearby_colleges_count": float(nearby_colleges_count),
    "nearby_hospitals_count": float(nearby_hospitals_count),
    "nearby_restaurants_count": float(nearby_restaurants_count),
    "nearby_parks_count": float(nearby_parks_count),
    "distance_to_nearest_brand_chai": float(distance_to_nearest_brand_chai),
    "rent_estimate": float(rent_estimate),
    "avg_income_area": float(avg_income_area)
}

input_df = pd.DataFrame([input_dict], columns=feature_cols)  # enforce exact column order

if debug:
    st.subheader("Debug: raw input dataframe")
    st.write(input_df)

# -------------------------
# Preprocessing: handle NaNs, scale
# -------------------------
# (Shouldn't be NaN because of widgets, but keep consistent with training)
input_df = input_df.fillna(input_df.median(numeric_only=True))

# Scale using the saved scaler
X_scaled = scaler.transform(input_df.values)

# -------------------------
# Predict
# -------------------------
# Support models without predict_proba gracefully
if hasattr(model, "predict_proba"):
    prob = model.predict_proba(X_scaled)[0][1]  # probability of class 1 (success)
else:
    # fallback: use decision_function or predict and convert into 0/1
    try:
        dec = model.decision_function(X_scaled)
        # normalize decision scores to 0-1
        dec_min, dec_max = dec.min(), dec.max()
        prob = float((dec - dec_min) / (dec_max - dec_min + 1e-9))
        prob = prob.item() if hasattr(prob, "item") else float(prob)
    except Exception:
        pred = model.predict(X_scaled)[0]
        prob = float(pred)

score_pct = prob * 100

# -------------------------
# Display result
# -------------------------
st.markdown("---")
st.subheader("Prediction")
st.metric(label="Success probability", value=f"{score_pct:.2f}%")

# Simple interpretation
if score_pct >= 75:
    st.success("✅ High Success Potential — good location signals")
elif score_pct >= 50:
    st.info("⚠️ Moderate Success Potential — some strengths, some weaknesses")
else:
    st.warning("❌ Low Success Potential — consider improving footfall / pricing / visibility")

# Helpful explanation
with st.expander("Why this prediction? (feature contribution - approximate)"):
    # try to show feature importances if available
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False)
        st.table(imp_df)
    else:
        st.write("Model does not expose feature_importances_.")

# Show the final feature vector sent to the model (debug)
if debug:
    st.subheader("Debug: scaled vector sent to model")
    st.write(pd.DataFrame(X_scaled, columns=feature_cols))

st.write("")  # spacing
st.caption("Model and scaler loaded from /models. Make sure they were trained using the same feature order above.")
