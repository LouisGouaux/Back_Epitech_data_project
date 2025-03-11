import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv("./dataset/hospital_data.csv")
df["season"] = df["season"].astype("category").cat.codes
df['date'] = pd.to_datetime(df['date'])
df_total = df.groupby("date", as_index=False).max()
X = df_total[["is_weekend", "is_holiday", "epidemic_flag", "vacation_flag", "season"]].copy()

X["year"] = df_total["date"].dt.year
X["month"] = df_total["date"].dt.month
X["day"] = df_total["date"].dt.day
X["weekday"] = df_total["date"].dt.weekday
X["weekend"] = (X["weekday"] >= 5).astype(int)

Y = df_total[["admissions_urgent", "admissions_programmed", "beds_occupied", "beds_available"]]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False, random_state=42)
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1, max_depth=5)

multi_model = MultiOutputRegressor(xgb_model)
multi_model.fit(X_train, Y_train)
Y_pred = multi_model.predict(X_test)
Y_pred_df = pd.DataFrame(Y_pred, columns=Y.columns, index=Y_test.index)
for col in Y.columns:
    mae = mean_absolute_error(Y_test[col], Y_pred_df[col])
for col in Y.columns:
    mae = mean_absolute_error(Y_test[col], Y_pred_df[col])
    rmse = np.sqrt(mean_squared_error(Y_test[col], Y_pred_df[col]))
    print(f"{col} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    print(f"Moyenne des {col} : {df[col].mean():.2f}")