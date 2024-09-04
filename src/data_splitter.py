import pandas as pd
from sklearn.model_selection import train_test_split

def sticky_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Perform a sticky train-test split with a fixed random state to ensure
    consistent splits across different runs.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)

def save_split_data(X_train, X_test, y_train, y_test, output_dir="data/"):
    """
    Save the split data into CSV files.
    """
    X_train.to_csv(f"{output_dir}X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}y_test.csv", index=False)