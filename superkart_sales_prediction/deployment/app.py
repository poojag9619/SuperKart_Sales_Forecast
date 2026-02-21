# ============================================================
# app.py — SuperKart Sales Forecast Streamlit App
# Loads the XGBoost model from Hugging Face Model Hub and
# provides an interactive UI for real-time sales predictions.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

# Download and load the trained model
model_path = hf_hub_download(repo_id="poojag007/superkart-sale-prediction", filename="best_sales_prediction_model_v1.joblib")
model = joblib.load(model_path)

st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒", layout="wide")

st.title("🛒 SuperKart Sales Forecast")
st.markdown("Predict **Product Store Sales** using our tuned XGBoost model. Fill in the details below and click **Predict**.")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🥤 Product Details")
    product_weight = st.number_input("Product Weight (g)", min_value=1.0, max_value=50.0, value=12.0, step=0.5)
    sugar_content  = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    allocated_area = st.slider("Product Allocated Area Ratio", 0.0, 0.3, 0.05, 0.005)
    product_type   = st.selectbox("Product Type", [
        "Fruits and Vegetables", "Snack Foods", "Household", "Frozen Foods",
        "Dairy", "Canned", "Baking Goods", "Health and Hygiene",
        "Soft Drinks", "Meat", "Breads", "Hard Drinks",
        "Others", "Starchy Foods", "Breakfast", "Seafood"
    ])
    product_mrp    = st.number_input("Product MRP ($)", min_value=10.0, max_value=300.0, value=150.0, step=5.0)

with col2:
    st.subheader("🏪 Store Details")
    store_est_year = st.slider("Store Establishment Year", 1985, 2020, 2005)
    store_age      = 2024 - store_est_year
    store_size     = st.selectbox("Store Size", ["Small", "Medium", "High"])
    city_type      = st.selectbox("City Type", ["Tier 1", "Tier 2", "Tier 3"])
    store_type     = st.selectbox("Store Type", [
        "Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"
    ])

with col3:
    st.subheader("📊 Forecast Period")
    forecast_period = st.selectbox("Forecast Granularity", ["Weekly", "Monthly", "Quarterly", "Annual"])
    multiplier = {"Weekly": 1, "Monthly": 4.33, "Quarterly": 13, "Annual": 52}[forecast_period]
    st.info(f"Weekly prediction × {multiplier} = {forecast_period} forecast")

st.divider()

if st.button("🔮 Predict Sales", use_container_width=True, type="primary"):
    raw_input = {
        "Product_Weight": product_weight, "Product_Sugar_Content": sugar_content,
        "Product_Allocated_Area": allocated_area, "Product_Type": product_type,
        "Product_MRP": product_mrp, "Store_Size": store_size,
        "Store_Location_City_Type": city_type, "Store_Type": store_type, "Store_Age": store_age
    }
    for col in ["Product_Sugar_Content", "Product_Type", "Store_Size", "Store_Location_City_Type", "Store_Type"]:
        le  = encoders[col]
        val = raw_input[col]
        raw_input[col] = int(le.transform([val])[0]) if val in le.classes_ else 0

    input_df     = pd.DataFrame([raw_input])[feature_names]
    weekly_sales = float(model.predict(input_df)[0])
    forecast     = weekly_sales * multiplier

    r1, r2, r3 = st.columns(3)
    r1.metric("Weekly Sales",             f"$ {weekly_sales:,.2f}")
    r2.metric(f"{forecast_period} Sales", f"$ {forecast:,.2f}")
    r3.metric("Annual Sales",             f"$ {weekly_sales * 52:,.2f}")
    st.success(f"✅ Predicted {forecast_period} Sales: $ {forecast:,.2f}")

    with st.expander("📋 Input Summary"):
        st.dataframe(pd.DataFrame([raw_input]).T.rename(columns={0: "Value"}))

st.divider()
st.caption("SuperKart MLOps Pipeline · Powered by XGBoost + Hugging Face + Streamlit")
