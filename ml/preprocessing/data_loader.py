import pandas as pd
import os


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "datasets", "walmart")

TRAIN_PATH = os.path.join(DATASET_PATH, "train.csv")
FEATURES_PATH = os.path.join(DATASET_PATH, "features.csv")
STORES_PATH = os.path.join(DATASET_PATH, "stores.csv")


def load_and_merge_data():

    print("Loading datasets...")

    train_df = pd.read_csv(TRAIN_PATH)
    features_df = pd.read_csv(FEATURES_PATH)
    stores_df = pd.read_csv(STORES_PATH)

    print("Merging datasets...")

    merged_df = pd.merge(
        train_df,
        features_df,
        on=["Store", "Date", "IsHoliday"],
        how="left"
    )

    merged_df = pd.merge(
        merged_df,
        stores_df,
        on="Store",
        how="left"
    )

    print("Cleaning data...")

    merged_df["Date"] = pd.to_datetime(merged_df["Date"])

    merged_df.fillna(0, inplace=True)

    print("Data Loaded Successfully!")

    return merged_df


if __name__ == "__main__":

    data = load_and_merge_data()

    print(data.head())
    print(data.shape)