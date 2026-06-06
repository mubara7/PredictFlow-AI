import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def forecast_next_3_months(model, df, sample_input):
    """
    Next 3 months ka forecast generate karta hai
    Returns: DataFrame with Date and Predicted_Sales
    """
    
    # Current date find karna (sample se)
    current_year = sample_input['Year']
    current_month = sample_input['Month']
    
    # Next 3 months ki dates generate karna
    dates = []
    for i in range(1, 4):  # 1, 2, 3 months ahead
        month = current_month + i
        year = current_year
        if month > 12:
            month = month - 12
            year += 1
        
        # First day of month
        date = datetime(year, month, 1)
        dates.append(date.strftime('%Y-%m-%d'))
    
    # Predictions store karne ke liye list
    predictions = []
    
    # Har month ke liye prediction
    for i, date_str in enumerate(dates):
        # Sample input copy karna
        sample = sample_input.copy()
        
        # Month aur Year update karna
        sample['Month'] = datetime.strptime(date_str, '%Y-%m-%d').month
        sample['Year'] = datetime.strptime(date_str, '%Y-%m-%d').year
        
        # DataFrame banana (same format as training)
        features = pd.DataFrame([sample])
        
        # Prediction
        pred = model.predict(features)[0]
        predictions.append(round(pred, 2))
    
    # Result DataFrame
    result_df = pd.DataFrame({
        'Date': dates,
        'Predicted_Sales': predictions
    })
    
    return result_df


def predict_sales(model, sample_input):
    """Single prediction ke liye"""
    features = pd.DataFrame([sample_input])
    prediction = model.predict(features)[0]
    return round(prediction, 2)


# For testing
if __name__ == '__main__':
    # Load model for testing
    import pickle
    with open('forecast_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load data
    df = pd.read_csv('../datasets/walmart/train.csv')
    
    # Test sample
    sample = {
        "Store": 1,
        "Dept": 1,
        "IsHoliday": 0,
        "Temperature": 40,
        "Fuel_Price": 3.2,
        "CPI": 211,
        "Unemployment": 8,
        "Size": 151315,
        "Month": 5,
        "Year": 2026
    }
    
    print("Single Prediction:", predict_sales(model, sample))
    print("\nNext 3 Months Forecast:")
    print(forecast_next_3_months(model, df, sample))