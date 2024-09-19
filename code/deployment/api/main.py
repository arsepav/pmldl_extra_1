from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostRegressor
import pandas as pd
import joblib
import sklearn


app = FastAPI()

class InputData(BaseModel):
    year: int
    make: str
    body: str
    transmission: str
    state: str
    condition: int
    odometer: float
    color: str
    interior: str
    mmr: float
    saledate: int

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

cat_features = ['make', 'body', 'transmission', 'state', 'color', 'interior']
numeric_features = ['year', 'condition', 'odometer', 'mmr', 'saledate']


@app.post("/predict/")
def predict(data: InputData):
    input_df = pd.DataFrame([data.dict()])

    X = input_df[['year', 'make', 'body', 'transmission', 'state', 'condition', 'odometer', 'color', 'interior', 'mmr', 'saledate']]

    X[numeric_features] = scaler.transform(X[numeric_features])

    try:
        prediction = model.predict(X)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

