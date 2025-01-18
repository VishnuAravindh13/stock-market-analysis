from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
from data_preprocessing import load_data, preprocess_data, split_data

def train_model(X_train, y_train):
    """Train a Random Forest model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.ravel())
    return model

if __name__ == "__main__":
    # Load and preprocess data
    data = load_data('data/stock_data.csv')
    data = preprocess_data(data)
    X_train, X_test, y_train, y_test, scaler = split_data(data)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Save the model and scaler
    joblib.dump(model, 'models/stock_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
