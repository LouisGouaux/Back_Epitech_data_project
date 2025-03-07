import pandas as pd
import matplotlib.pyplot as plt
from main import import_dataset
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

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


def sarima_model(df):
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    df_copy.set_index("date", inplace=True)

    df_daily = df_copy.groupby("date")["admissions_urgent"].sum()
    df_daily.fillna(method="ffill", inplace=True)
    p, d, q = (2, 1, 2)
    P, D, Q, s = (1, 1, 1, 7)
    model = SARIMAX(df_daily, order=(p, d, q), seasonal_order=(P, D, Q, s))
    result = model.fit()
    future = result.get_forecast(steps=30)
    future_conf = future.conf_int()
    df_futur = pd.DataFrame({
        "Date": future_conf.index,
        "Prévision": future.predicted_mean,
        "Borne inférieure": future_conf.iloc[:, 0],
        "Borne supérieure": future_conf.iloc[:, 1]
    })
    print(df_futur)

    plt.figure(figsize=(12, 6))
    plt.plot(df_daily, label="Données réelles")
    plt.plot(future.predicted_mean, label="Prévisions (SARIMA)", linestyle="dashed")
    plt.fill_between(future_conf.index, future_conf.iloc[:, 0], future_conf.iloc[:, 1], color='pink', alpha=0.3)
    plt.legend()
    plt.show()
    return df_futur
sarima_model(import_dataset())