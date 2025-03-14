import datetime
from jours_feries_france import JoursFeries
from vacances_scolaires_france import SchoolHolidayDates
from main import import_dataset
import pandas as pd
from calendar import monthrange
import numpy as np
import xgboost as xgb
import os

df = import_dataset()
models = {}
services = df['service'].unique()
for service in services:
    models[service] = {}
    for i in ["admissions_urgent", "admissions_programmed", "beds_occupied", "beds_available"]:
        model_path = f"models/xgboost_model_{service}_{i}.json"
        if os.path.exists(model_path):
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            models[service][i] = model
            print(f"Modèle chargé : {model_path}")
        else:
            print(f"Modèle introuvable : {model_path}")

def convert_to_serializable(obj):
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    return obj


def create_futur_data(date):
    df = import_dataset()
    services = df['service'].unique()
    year = pd.to_datetime(date).year

    res = JoursFeries.for_year(int(year))
    res = pd.to_datetime(list(res.values())).to_numpy()
    d = SchoolHolidayDates()
    holidays = d.holidays_for_year_and_zone(int(year), 'C')
    holidays = pd.to_datetime(list(holidays)).to_numpy()

    day_data = pd.DataFrame([{
        "is_weekend": pd.to_datetime(date).weekday() >= 5,
        "is_holiday": 1 if (np.datetime64(date) in res or np.datetime64(date) in holidays) else 0,
        "epidemic_flag": 0,
        "vacation_flag": 0,
        "season": 1 if pd.to_datetime(date).month in [12, 1, 2] else 2 if pd.to_datetime(date).month in [3, 4,
                                                                                                         5] else 3 if pd.to_datetime(
            date).month in [6, 7, 8] else 4,
        "year": year,
        "month": pd.to_datetime(date).month,
        "day": pd.to_datetime(date).day,
        "weekday": pd.to_datetime(date).weekday(),
    }])

    day_data["weekday"] = day_data["weekday"].astype(int)

    resource_columns = [
        "beds_occupied", "beds_available", "staff_medecin", "staff_chirurgien", "staff_infirmier",
        "staff_aide_soignant", "staff_sage_femme", "staff_psychologue", "staff_admin", "staff_astreinte",
        "medicaments_service", "epi_service", "materiel_chirurgical_service", "equipements_biomedicaux_service",
        "produits_hygiene_service", "produits_pharmaceutiques_service", "vaccins_service"
    ]

    futur_resources_list = []

    df_numeric = df.select_dtypes(include=[np.number]).copy()
    df_numeric["service"] = df["service"]
    mean_values = df_numeric.groupby("service").mean().round(0).to_dict()

    for service in services:
        row = {}

        if service in models:
            predicted_columns = models[service]["beds_occupied"].get_booster().feature_names
            print(f"Colonnes prédictibles pour {service} :", predicted_columns)

            for target in models[service]:
                row[target] = models[service][target].predict(day_data)[0].round(0)

        else:
            print(f"Modèle introuvable pour {service}, utilisation des moyennes.")

        row["service"] = service
        for col in resource_columns:
            if col not in row:
                row[col] = mean_values.get(col, {}).get(service, 0)

        row["service"] = service
        futur_resources_list.append(row)

    df_futur = pd.DataFrame(futur_resources_list)

    columns_to_remove = ["is_weekend", "season", "year", "month", "day", "weekday"]
    df_futur.drop(columns=[col for col in columns_to_remove if col in df_futur.columns], errors="ignore", inplace=True)

    total_row = df_futur.sum().to_dict()
    total_row["service"] = "Total"
    for row in futur_resources_list:
        for key in row:
            row[key] = convert_to_serializable(row[key])

    for key in total_row:
        total_row[key] = convert_to_serializable(total_row[key])

    futur_resources_list.append(total_row)

    return futur_resources_list


def get_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


def create_futur_anual_calendar(year):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    future_calendar_list = []
    res = JoursFeries.for_year(int(year))
    res = pd.to_datetime(list(res.values())).to_numpy()
    d = SchoolHolidayDates()
    holidays = d.holidays_for_year_and_zone(int(year), 'C')
    holidays = pd.to_datetime(list(holidays)).to_numpy()

    for date in date_range:
        day_data = {
            "date": date.strftime("%Y-%m-%d"),
            "day_of_week": date.strftime("%A"),
            "is_weekend": date.weekday() >= 5,
            "season": get_season(date),
            "is_holiday": 1 if (np.datetime64(date) in res or np.datetime64(date) in holidays) else 0,
            "vacation_flag": 0,
            "epidemic_flag": 0,
            "canicule_flag": 0,
            "plan_blanc_flag": 0
        }
        future_calendar_list.append(day_data)

    return future_calendar_list


def create_futur_monthly_calendar(year, month):
    start_date = f"{year}-{month}-01"
    last_day = monthrange(int(year), int(month))[1]
    end_date = f"{year}-{month}-{last_day}"
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    future_calendar_list = []
    res = JoursFeries.for_year(int(year))
    res = pd.to_datetime(list(res.values())).to_numpy()
    d = SchoolHolidayDates()
    holidays = d.holidays_for_year_and_zone(int(year), 'C')
    holidays = pd.to_datetime(list(holidays)).to_numpy()

    for date in date_range:
        day_data = {
            "date": date.strftime("%Y-%m-%d"),
            "day_of_week": date.strftime("%A"),
            "is_weekend": date.weekday() >= 5,
            "season": get_season(date),
            "is_holiday": 1 if (np.datetime64(date) in res or np.datetime64(date) in holidays) else 0,
            "vacation_flag": 0,
            "epidemic_flag": 0,
            "canicule_flag": 0,
            "plan_blanc_flag": 0
        }
        future_calendar_list.append(day_data)

    return future_calendar_list
