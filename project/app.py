import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# =========================================================
# 1. DATASET - Dataset already موجود ہے اسی code کے اندر
# =========================================================

data = {
    "Bedrooms": [
        2, 3, 3, 4, 4, 5, 2, 3, 4, 5,
        3, 4, 5, 2, 3, 4, 5, 3, 4, 5,
        2, 3, 4, 5, 3, 4, 5, 2, 4, 3
    ],

    "Bathrooms": [
        1, 2, 2, 3, 3, 4, 1, 2, 3, 4,
        2, 3, 4, 1, 2, 3, 4, 2, 3, 4
        ,1, 2, 3, 4, 2, 3, 4, 1, 3, 2
    ],

    "Area_sqft": [
        800, 1200, 1500, 1800, 2200, 2800, 750, 1300, 2000, 3000,
        1400, 1900, 2600, 700, 1100, 2100, 3200, 1250, 2300, 3500,
        900, 1450, 2500, 3300, 1350, 2000, 2900, 850, 2400, 1600
    ],

    "House_Age": [
        10, 5, 3, 8, 4, 2, 15, 6, 5, 3,
        7, 4, 1, 20, 8, 6, 2, 9, 5, 1,
        12, 6, 4, 2, 7, 5, 3, 15, 4, 8
    ],

    "Floors": [
        1, 2, 2, 2, 2, 3, 1, 2, 2, 3,
        2, 2, 3, 1, 2, 2, 3, 2, 2, 3,
        1, 2, 2, 3, 2, 2, 3, 1, 2, 2
    ],

    "Parking": [
        1, 1, 2, 2, 2, 3, 0, 1, 2, 3,
        1, 2, 3, 0, 1, 2, 3, 1, 2, 3,
        1, 1, 2, 3, 1, 2, 3, 0, 2, 1
    ],

    "Location": [
        "Multan", "Lahore", "Islamabad", "Lahore", "Islamabad",
        "Lahore", "Multan", "Faisalabad", "Islamabad", "Lahore",
        "Multan", "Rawalpindi", "Islamabad", "Multan", "Faisalabad",
        "Lahore", "Islamabad", "Rawalpindi", "Lahore", "Islamabad",
        "Multan", "Faisalabad", "Lahore", "Islamabad", "Rawalpindi",
        "Lahore", "Islamabad", "Multan", "Rawalpindi", "Faisalabad"
    ],

    "Price": [
        4500000, 7200000, 9500000, 11000000, 13500000,
        18000000, 4000000, 6500000, 12500000, 19000000,
        7800000, 11500000, 17500000, 3500000, 6000000,
        12000000, 21000000, 7000000, 14500000, 22000000,
        4800000, 7600000, 14000000, 23000000, 7200000,
        13000000, 19500000, 4200000, 15000000, 8500000
    ]
}


# Dataset کو DataFrame میں تبدیل کرنا
df = pd.DataFrame(data)


# =========================================================
# 2. MACHINE LEARNING MODEL
# =========================================================

X = df.drop("Price", axis=1)
y = df["Price"]


# Training اور testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Numeric columns
numeric_features = [
    "Bedrooms",
    "Bathrooms",
    "Area_sqft",
    "House_Age",
    "Floors",
    "Parking"
]


# Categorical column
categorical_features = ["Location"]


# Location کو numbers میں convert کرنا
preprocessor = ColumnTransformer(
    transformers=[
        ("location", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ],
    remainder="passthrough"
)


# Random Forest ML Model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        ("regressor", RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=42
        ))
    ]
)


# Model training
model.fit(X_train, y_train)


# =========================================================
# 3. MODEL TESTING
# =========================================================

test_prediction = model.predict(X_test)

mae = mean_absolute_error(y_test, test_prediction)
r2 = r2_score(y_test, test_prediction)


# =========================================================
# 4. STREAMLIT WEBSITE
# =========================================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)


st.title("🏠 House Price Prediction")
st.write(
    "Machine Learning کی مدد سے گھر کی estimated price معلوم کریں۔"
)


st.divider()


# =========================================================
# 5. USER INPUT
# =========================================================

st.subheader("🏡 Enter House Details")


bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)


bathrooms = st.number_input(
    "Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)


area = st.number_input(
    "Area (Square Feet)",
    min_value=300,
    max_value=10000,
    value=1500,
    step=50
)


age = st.number_input(
    "House Age (Years)",
    min_value=0,
    max_value=100,
    value=5
)


floors = st.number_input(
    "Floors",
    min_value=1,
    max_value=5,
    value=2
)


parking = st.number_input(
    "Parking Spaces",
    min_value=0,
    max_value=5,
    value=1
)


location = st.selectbox(
    "Location",
    [
        "Multan",
        "Lahore",
        "Islamabad",
        "Rawalpindi",
        "Faisalabad"
    ]
)


st.divider()


# =========================================================
# 6. PREDICTION
# =========================================================

if st.button("🔮 Predict House Price"):

    new_house = pd.DataFrame([{
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Area_sqft": area,
        "House_Age": age,
        "Floors": floors,
        "Parking": parking,
        "Location": location
    }])


    prediction = model.predict(new_house)[0]


    st.success(
        f"🏠 Estimated House Price: Rs. {prediction:,.0f}"
    )


    st.info(
        "یہ prediction اس code کے اندر موجود sample dataset "
        "پر trained Machine Learning model سے حاصل کی گئی ہے۔"
    )


# =========================================================
# 7. MODEL PERFORMANCE
# =========================================================

with st.expander("📊 Model Performance"):

    st.write(f"**R² Score:** {r2:.2f}")

    st.write(f"**Mean Absolute Error:** Rs. {mae:,.0f}")

    st.write(
        "Random Forest Regression استعمال کیا گیا ہے۔"
    )


# =========================================================
# 8. DATASET PREVIEW
# =========================================================

with st.expander("📋 View Dataset"):

    st.dataframe(df)


st.divider()

st.caption(
    "⚠️ This is a Machine Learning demonstration using sample data. "
    "It should not be used as an actual property valuation."
)