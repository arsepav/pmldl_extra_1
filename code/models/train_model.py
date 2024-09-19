import mlflow
import mlflow.sklearn
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import joblib

mlflow.set_tracking_uri("http://localhost:5000")

cat_features=['make', 'body', 'transmission', 'state', 'color', 'interior']
numeric_features=['year', 'condition', 'odometer', 'mmr', 'saledate']

train_data_path = "data/processed/train_data.csv"
test_data_path = "data/processed/test_data.csv"


def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data


def feature_engineering(data, scaler = None):
    print(1)
    if not scaler:
        scaler = StandardScaler()
        
        data[numeric_features] = scaler.fit_transform(data[numeric_features])
    else:
        data[numeric_features] = scaler.transform(data[numeric_features])

    return data, scaler


def train_model(X_train, y_train):
    model = CatBoostRegressor(cat_features=cat_features)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    # Предсказания модели
    y_pred = model.predict(X_test)
    
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, mae, r2


def log_metrics_and_model(model, scaler, mse, mae, r2, X_example):
    with mlflow.start_run():
        mlflow.log_metric("mean_squared_error", mse)
        mlflow.log_metric("mean_absolute_error", mae)
        mlflow.log_metric("r2_score", r2)
        
        mlflow.sklearn.log_model(model, "model", input_example=X_example)

        scaler_path = "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path="scaler")
        
        os.remove(scaler_path)


if __name__ == "__main__":
    train_data, test_data = load_data(train_data_path, test_data_path)
    
    X_train, y_train = train_data.drop(columns=['sellingprice']), train_data['sellingprice']
    X_test, y_test = test_data.drop(columns=['sellingprice']), test_data['sellingprice']
    
    X_train, scaler = feature_engineering(X_train)
    X_test, _ = feature_engineering(X_test, scaler)

    print(X_train.head())

    print(X_test.head())
    
    model = train_model(X_train, y_train)
    

    mse, mae, r2 = evaluate_model(model, X_test, y_test)
    

    example_input = X_test.iloc[0].values.reshape(1, -1)
    log_metrics_and_model(model, scaler, mse, mae, r2, example_input)
    
    print("Model trained, evaluated, and saved successfully.")
