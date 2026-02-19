import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Manufacturing Output Predictor", layout="wide")

st.title("🏭 Manufacturing Output Prediction")
st.write("Model trained using manufacturing_data.csv")

# =====================================================
# LOAD DATASET
# =====================================================

try:
    data = pd.read_csv("C:/Users/Personal/Desktop/Capstone p1/Dataset_manufacturing_1000_samples .csv")
except FileNotFoundError:
    st.error("❌ manufacturing_data.csv not found.")
    st.stop()

# =====================================================
# KEEP ONLY NUMERIC COLUMNS
# =====================================================

data = data.select_dtypes(include=[np.number])

# =====================================================
# CHECK TARGET COLUMN
# =====================================================

if "Parts_Per_Hour" not in data.columns:
    st.error("❌ 'Parts_Per_Hour' column missing or not numeric.")
    st.stop()

# =====================================================
# REPLACE INFINITE VALUES
# =====================================================

data = data.replace([np.inf, -np.inf], np.nan)

# =====================================================
# DROP ROWS WHERE TARGET IS MISSING
# =====================================================

data = data.dropna(subset=["Parts_Per_Hour"])

# =====================================================
# SPLIT FEATURES & TARGET
# =====================================================

X = data.drop("Parts_Per_Hour", axis=1)
y = data["Parts_Per_Hour"]

# =====================================================
# IMPUTE MISSING FEATURE VALUES
# =====================================================

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# =====================================================
# SCALE FEATURES
# =====================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =====================================================
# TRAIN MODEL
# =====================================================

model = LinearRegression()
model.fit(X_train, y_train)

# =====================================================
# MODEL EVALUATION
# =====================================================

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)
col1.metric("R² Score", round(r2, 3))
col2.metric("MSE", round(mse, 2))
col3.metric("RMSE", round(rmse, 2))
col4.metric("MAE", round(mae, 2))

# =====================================================
# PREDICTION SECTION
# =====================================================

st.subheader("🔧 Enter Machine Parameters")

input_values = []

for col in X.columns:
    value = st.number_input(
        f"{col}",
        float(X[col].min()),
        float(X[col].max()),
        float(X[col].mean())
    )
    input_values.append(value)

if st.button("Predict Output 🚀"):

    input_array = np.array([input_values])

    # Apply same preprocessing steps
    input_imputed = imputer.transform(input_array)
    input_scaled = scaler.transform(input_imputed)

    prediction = model.predict(input_scaled)

    st.success(f"✅ Predicted Output: {prediction[0]:.2f} Parts per Hour")

    if prediction[0] > y.mean():
        st.info("High Production Efficiency 📈")
    else:
        st.warning("Below Average Production ⚠️")

st.markdown("---")
st.markdown("Developed using Streamlit | Linear Regression | Manufacturing Optimization")
