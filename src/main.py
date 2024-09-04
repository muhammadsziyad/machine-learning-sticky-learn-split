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