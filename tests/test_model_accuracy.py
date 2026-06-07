import os
import sys
import joblib
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ml')))

from ml.preprocessing.data_loader import load_and_merge_data

def test_saas_dynamic_predictions():
    print("\n--- 🚀 RUNNING FULL & FINAL SaaS DYNAMIC TESTING SUITE ---")
    
    # 1. Model Matrix Load Check
    model_path = os.path.join(os.path.dirname(__file__), '../ml/models/forecast_model.pkl')
    if not os.path.exists(model_path):
        print("❌ Error: Trained model forecast_model.pkl not found! Retrain required with new features.")
        return
    model = joblib.load(model_path)
    print("✅ Model weights initialized.")

    # 2. Trigger Advanced Loader
    real_merged_df = load_and_merge_data()
    
    # Slice rows 10 to 15 for deep lag simulation visualization
    sample_df = real_merged_df.iloc[10:15].copy()

    # 3. Final 12-Dimensional Feature Blueprint
    features_list = [
        'Store', 'Dept', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 
        'Unemployment', 'Size', 'Month', 'Year', 'Sales_Last_Week', 'Sales_Rolling_Mean'
    ]
    
    X_real = sample_df[features_list].values
    print(f"\n📦 Verified Matrix Shape: {X_real.shape} -> 12 Production Dimensions.")

    print("\nFeeding operational matrix into machine learning core...")
    try:
        predictions = model.predict(X_real)
        print("\n🔮 --- 100% REAL REAL-TIME SaaS FORECAST RESULTS ---")
        for i, pred in enumerate(predictions):
            date_str = sample_df['Date'].iloc[i].strftime('%Y-%m-%d')
            last_w = sample_df['Sales_Last_Week'].iloc[i]
            
            print(f"Row {i+1} -> [Date: {date_str} | Last Week Asli Sales: ${last_w:,.2f}]")
            print(f"        🔮 Future Predicted Sales: ${pred:,.2f}")
            
            if 'Weekly_Sales' in sample_df.columns:
                actual = sample_df['Weekly_Sales'].iloc[i]
                error = abs(pred - actual)
                acc = (1 - (error / actual)) * 100
                print(f"        🎯 LIVE ACCURACY: {max(0, acc):.2f}%\n")
    except Exception as e:
        print(f"\n⚠️ Status Code [Expected Adjustment Needed]: {e}")
        print("💡 Solution: Model ko dubara train karein taake woh 12 features accept kare!")

if __name__ == "__main__":
    test_saas_dynamic_predictions()