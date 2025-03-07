import pandas as pd
import matplotlib.pyplot as plt
from main import import_dataset
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose

df = import_dataset()


def stationary_test(df, value):
    if (value not in df.columns):
        return f"Erreur : la Colonne \"{value}\" n'existe pas dans le dataframe."
    if df[value].isnull().sum() > 0:
        return f"Erreur : La colonne \"{value}\" contient des valeurs manquantes."

    result = adfuller(df[value])
    stats = {
        'Test Statistic': f"{result[0]:.4f}",
        'P-Value': f"{result[1]:.4f}",
        'Critical Values': {key: f"{val:.4f}" for key, val in result[4].items()}
    }
    stats['conclusion'] = "La série est stationnaire." if result[1] < 0.05 else "La série n'est pas stationnaire."
    return (stats)


def seasonality_test(df, period=7):
    df_daily = df.copy()
    df_daily['date'] = pd.to_datetime(df['date'])
    df_daily.set_index("date", inplace=True)
    df_daily = df.groupby("date")["admissions_urgent"].sum()
    decomposition = seasonal_decompose(df_daily, model="additive", period=period)
    plt.figure(figsize=(12, 8))
    decomposition.plot()
    plt.show()
seasonality_test(df)