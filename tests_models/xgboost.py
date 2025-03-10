import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv("./dataset/hospital_data.csv")
df['date'] = pd.to_datetime(df['date'])
df_total = df.groupby("date", as_index=False).sum()
X = df_total[["date", "day_of_week", "is_weekend", "is_holiday", "epidemic_flag", "vacation_flag", "season"]].copy()

X["year"] = df_total["date"].dt.year
X["month"] = df_total["date"].dt.month
X["day"] = df_total["date"].dt.day
X["weekday"] = df_total["date"].dt.weekday
X["weekend"] = (X["weekday"] >= 5).astype(int)
X.drop(columns=["weekday"], inplace=True)

print(X.head())