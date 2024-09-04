# Sticky Learn Split data for Train and Validation

In Python, a "sticky learn split" typically means ensuring that your data splits (train, validation, test) are consistent across multiple runs or across different parts of the workflow. This is often achieved by using a fixed random seed or ensuring the same splits are reused in different parts of your code. Here's how you can set up such a project:

Step 1: Set Up Your Python Environment
1. Create a Virtual Environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Or we can use conda for alternative virtual environtment

2. Install Required Libraries: Install the necessary libraries:

```bash
pip install numpy pandas scikit-learn
```

Step 2: Create the Project Structure
Organize your project with the following structure:

```css
sticky-learn-split/
│
├── data/
│   └── your_dataset.csv
├── src/
│   ├── data_splitter.py
│   └── main.py
├── venv/
├── .gitignore
├── requirements.txt
└── README.md
```

Step 3: Implement Sticky Data Splitting
1. data_splitter.py: This module handles the splitting of your dataset while ensuring consistent splits across different runs.

```python
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
```

2. main.py: This is the main script to load the data, split it, and save the results.

```Python
from src.data_splitter import sticky_train_test_split, load_data, save_split_data

def main():
    # Load the dataset
    data = load_data('data/your_dataset.csv')
    
    # Assuming the last column is the target variable
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Split the data
    X_train, X_test, y_train, y_test = sticky_train_test_split(X, y)

    # Save the split data
    save_split_data(X_train, X_test, y_train, y_test)

    print("Data has been split and saved successfully.")

if __name__ == "__main__":
    main()
```


Step 4: Run the Project
1. Run the Script: Run the main.py script to split the data and save the splits.

```bash
python src/main.py
```

2. Check the Output: After running the script, you should find the split datasets (X_train.csv, X_test.csv, y_train.csv, y_test.csv) saved in the data/ directory.

Step 5: (Optional) Version Control and Documentation
1. Add a .gitignore File: To ignore unnecessary files, such as the virtual environment and data files, create a .gitignore file:

Summary
You've now created a Python project that splits data into training and testing sets in a consistent (sticky) manner, ensuring reproducibility across different runs. This setup is useful when you're dealing with machine learning tasks where consistent data splitting is crucial.