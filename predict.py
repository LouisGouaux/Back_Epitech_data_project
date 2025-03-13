import datetime
from jours_feries_france import JoursFeries
from vacances_scolaires_france import SchoolHolidayDates
from main import import_dataset
import pandas as pd
from calendar import monthrange
import numpy as np



def create_futur_data(date):
    df = import_dataset()
    services = df['service'].unique()
    resource_columns = [
        "beds_occupied", "beds_available", "staff_medecin", "staff_chirurgien", "staff_infirmier",
        "staff_aide_soignant", "staff_sage_femme", "staff_psychologue", "staff_admin", "staff_astreinte",
        "medicaments_service", "epi_service", "materiel_chirurgical_service", "equipements_biomedicaux_service",
        "produits_hygiene_service", "produits_pharmaceutiques_service", "vaccins_service"
    ]
    df_mean = df.groupby("service")[resource_columns].mean()
    futur_resources_list = []
    for service in services:
        row = df_mean.loc[service].to_dict()
        row["service"] = service
        row["date"] = date
        futur_resources_list.append(row)
    total_row = pd.DataFrame(futur_resources_list)[resource_columns].sum().to_dict()
    total_row["service"] = "Total"
    total_row["date"] = date
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
                "is_holiday": 1 if (np.datetime64(date) in res or np.datetime64(date)  in holidays) else 0,
                "vacation_flag": 0,
                "epidemic_flag": 0,
                "canicule_flag": 0,
                "plan_blanc_flag": 0
            }
            future_calendar_list.append(day_data)

        return future_calendar_list

