import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "walmart")

TRAIN_PATH = os.path.join(DATASET_PATH, "train.csv")
FEATURES_PATH = os.path.join(DATASET_PATH, "features.csv")
STORES_PATH = os.path.join(DATASET_PATH, "stores.csv")

def load_and_merge_data():
    print("📋 [SaaS Engine] Loading raw datasets...")
    train_df = pd.read_csv(TRAIN_PATH)
    features_df = pd.read_csv(FEATURES_PATH)
    stores_df = pd.read_csv(STORES_PATH)

    print("🔗 [SaaS Engine] Merging datasets on operational keys...")
    merged_df = pd.merge(train_df, features_df, on=["Store", "Date", "IsHoliday"], how="left")
    merged_df = pd.merge(merged_df, stores_df, on="Store", how="left")

    print("🧹 [SaaS Engine] Cleaning and sorting data timelines...")
    merged_df["Date"] = pd.to_datetime(merged_df["Date"])
    # Sort karna zaroori hai taake pichle hafte ki sales hamesha sahi order mein aayein
    merged_df = merged_df.sort_values(by=["Store", "Dept", "Date"]).reset_index(drop=True)

    print("📊 [SaaS Engine] Engineering Advanced Time-Series Lag Features...")
    # 1. Core Feature Engineering: Last Week Sales (Shift 1 Week)
    merged_df['Sales_Last_Week'] = merged_df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1)
    
    # 2. Rolling Mean: Pichle 4 hafton ka trend average
    merged_df['Sales_Rolling_Mean'] = merged_df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
        lambda x: x.shift(1).rolling(window=4, min_periods=1).mean()
    )

    # 3. Extracting Numeric Calendar Profiles
    merged_df['Month'] = merged_df['Date'].dt.month
    merged_df['Year'] = merged_df['Date'].dt.year

    # Handling initial rows missing lags (Null points ko baseline media se fill karte hain)
    merged_df['Sales_Last_Week'] = merged_df['Sales_Last_Week'].fillna(merged_df['Weekly_Sales'])
    merged_df['Sales_Rolling_Mean'] = merged_df['Sales_Rolling_Mean'].fillna(merged_df['Weekly_Sales'])
    merged_df.fillna(0, inplace=True)

    print("✅ [SaaS Engine] Pure Production Matrix Ready!")
    return merged_df

if __name__ == "__main__":
    data = load_and_merge_data()
    print(f"\nSample Matrix Preview:\n{data[['Date', 'Weekly_Sales', 'Sales_Last_Week', 'Sales_Rolling_Mean']].head(10)}")
    print(f"Matrix Dimensions: {data.shape}")