import pandas as pd
from main import import_dataset
from statsmodels.tsa.stattools import adfuller, acf, pacf

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
    stats['conclusion'] = "La série est stationnaire." if result[1]<0.05 else "La série n'est pas stationnaire."
    return (stats)


