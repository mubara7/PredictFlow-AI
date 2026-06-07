import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "forecast_model.pkl")

# Ensure models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Strict 12 features blueprint for SaaS
FEATURES_LIST = [
    'Store', 'Dept', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 
    'Unemployment', 'Size', 'Month', 'Year', 'Sales_Last_Week', 'Sales_Rolling_Mean'
]

def train_production_model():
    """Real dynamic features par model ko train aur save karta hai"""
    print("\n🏋️‍♂️ [SaaS Engine] Initializing Model Retraining on 12 Features...")
    
    # Import inside function to avoid circular import issues
    from ml.preprocessing.data_loader import load_and_merge_data
    
    # Load pure engineered dataset (with 12 columns)
    df_real = load_and_merge_data()
    
    X = df_real[FEATURES_LIST].values
    y = df_real['Weekly_Sales'].values
    
    print(f"🔥 Training RandomForest on Matrix Shape: {X.shape}")
    
    # Fast training configuration for testing
    model = RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Save the smart weights with feature names to suppress warnings
    model.feature_names_in_ = np.array(FEATURES_LIST)
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Saved updated 12-feature model weights to: {MODEL_PATH}")
    return model

def forecast_next_3_months(model, sample_input):
    """Dynamic multi-month auto-regressive forecast engine"""
    current_year = sample_input['Year']
    current_month = sample_input['Month']
    
    dates = []
    for i in range(1, 4):
        month = current_month + i
        year = current_year
        if month > 12:
            month = month - 12
            year += 1
        dates.append(datetime(year, month, 1).strftime('%Y-%m-%d'))
    
    predictions = []
    current_lag_sales = sample_input['Sales_Last_Week']
    current_rolling_mean = sample_input['Sales_Rolling_Mean']
    
    for date_str in dates:
        sample = sample_input.copy()
        sample['Month'] = datetime.strptime(date_str, '%Y-%m-%d').month
        sample['Year'] = datetime.strptime(date_str, '%Y-%m-%d').year
        sample['Sales_Last_Week'] = current_lag_sales
        sample['Sales_Rolling_Mean'] = current_rolling_mean
        
        features_df = pd.DataFrame([sample])[FEATURES_LIST]
        
        # Predict using feature names to avoid user warnings
        pred = model.predict(features_df)[0]
        predictions.append(round(pred, 2))
        
        # Feedback loop logic for true dynamic variance
        current_lag_sales = pred
        current_rolling_mean = (current_rolling_mean + pred) / 2
    
    return pd.DataFrame({'Date': dates, 'Predicted_Sales': predictions})

if __name__ == '__main__':
    # Force retraining to align 12 feature spaces
    model = train_production_model()
    
    # 100% Real Verification Sample
    sample = {
        "Store": 1, "Dept": 1, "IsHoliday": 0, "Temperature": 42.31,
        "Fuel_Price": 2.852, "CPI": 211.231, "Unemployment": 7.905,
        "Size": 151315, "Month": 2, "Year": 2010,
        "Sales_Last_Week": 24924.50, "Sales_Rolling_Mean": 24924.50
    }
    
    print("\n--- 📊 EXECUTING FORECAST ENGINE ---")
    sample_df = pd.DataFrame([sample])[FEATURES_LIST]
    single_pred = model.predict(sample_df)[0]
    print(f"Single Prediction for Row 1: ${single_pred:,.2f}")
    
    print("\nNext 3 Months Advanced Forecast Layout:")
    print(forecast_next_3_months(model, sample))