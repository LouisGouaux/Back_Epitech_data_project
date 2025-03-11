import pandas as pd
import os
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

os.makedirs("models", exist_ok=True)

df = pd.read_csv("./dataset/hospital_data.csv")
df["season"] = df["season"].astype("category").cat.codes
df['date'] = pd.to_datetime(df['date'])
services = df["service"].unique()
errors = {}
for service in services:
    print (f"Entraînement pour le service {service}… \n")
    df_service = df[df["service"] == service].copy()

    X = df_service[["is_weekend", "is_holiday", "epidemic_flag", "vacation_flag", "season"]].copy()
    X["year"] = df_service["date"].dt.year
    X["month"] = df_service["date"].dt.month
    X["day"] = df_service["date"].dt.day
    X["weekday"] = df_service["date"].dt.weekday
    X.ffill(inplace=True)

    Y = df_service[["admissions_urgent", "admissions_programmed", "beds_occupied", "beds_available"]]
    Y.ffill(inplace=True)


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False, random_state=42)
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1, max_depth=5)

    multi_model = MultiOutputRegressor(xgb_model)
    multi_model.fit(X_train, Y_train)
    multi_model.estimators_[0].save_model(f"models/xgboost_model_{service}.json")

    Y_pred = multi_model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns=Y.columns, index=Y_test.index)
    for col in Y.columns:
        mae = mean_absolute_error(Y_test[col], Y_pred_df[col])
        rmse = np.sqrt(mean_squared_error(Y_test[col], Y_pred_df[col]))
        print(f"{col} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, Moyenne: {df_service[col].mean():.2f}")
