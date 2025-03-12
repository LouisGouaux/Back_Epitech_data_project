import datetime
from jours_feries_france import JoursFeries
from vacances_scolaires_france import SchoolHolidayDates
from main import import_dataset
import pandas as pd

res = JoursFeries.for_year(2026)

d = SchoolHolidayDates()
holidays = d.holidays_for_year(2018)


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
