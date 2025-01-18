import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """Load stock market data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Clean and preprocess the data."""
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data = data.dropna()  # Remove missing values
    return data

def split_data(data):
    """Split data into training and testing sets."""
    X = data[['Open', 'High', 'Low', 'Volume']].values
    y = data['Close'].values
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y

if __name__ == "__main__":
    data = load_data('data/stock_data.csv')
    data = preprocess_data(data)
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = split_data(data)
    print(f"Training data: {X_train.shape}, Testing data: {X_test.shape}")
