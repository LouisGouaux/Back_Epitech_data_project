import pandas as pd
import matplotlib.pyplot as plt


def import_dataset():
    df = pd.read_csv("./dataset/hospital_data.csv")
    return (df)


def get_resources(date):
    df = import_dataset()
    df_filtered = df[df["date"] == date]
    resource_columns = [
        "service", "beds_occupied", "beds_available", "staff_medecin", "staff_chirurgien", "staff_infirmier",
        "staff_aide_soignant", "staff_sage_femme", "staff_psychologue", "staff_admin", "staff_astreinte",
        "medicaments_service", "epi_service", "materiel_chirurgical_service", "equipements_biomedicaux_service",
        "produits_hygiene_service", "produits_pharmaceutiques_service", "vaccins_service"
    ]
    total_row = df_filtered[resource_columns].sum().to_dict()
    total_row['service'] = 'Total'
    total_row['date'] = date
    resources_list = df_filtered[["date", "service"] + resource_columns].to_dict(orient="records")
    resources_list.append(total_row)
    return resources_list


def get_flags_by_year(year):
    df = import_dataset()
    df_filtered = df.copy()
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    df_filtered = df_filtered[df_filtered["date"].dt.year == int(year)]
    flag_columns = ["day_of_week", "is_weekend", "is_holiday",
                    "vacation_flag", "season", "epidemic_flag", "canicule_flag", "plan_blanc_flag"]
    df_grouped = df_filtered.groupby("date")[flag_columns].max().reset_index()
    df_grouped["date"] = df_grouped["date"].astype(str)
    return df_grouped.to_dict(orient="records")


def get_flags_by_month(year, month):
    df = import_dataset()
    df["date"] = pd.to_datetime(df["date"])
    df_filtered = df[(df["date"].dt.year == int(year)) & (df["date"].dt.month == int(month))].copy()
    flag_columns = ["day_of_week", "is_weekend", "is_holiday", "vacation_flag",
                    "season", "epidemic_flag", "canicule_flag", "plan_blanc_flag"]
    df_grouped = df_filtered.groupby("date")[flag_columns].max().reset_index()
    df_grouped["date"] = df_grouped["date"].astype(str)
    return df_grouped.to_dict(orient="records")


def get_aggregated_data(aggregate_column, value_order=None, value_columns=None):
    df = import_dataset()
    if value_columns is None:
        value_columns = ["admissions_urgent", "admissions_programmed"]

    if aggregate_column not in df.columns:
        raise ValueError(f"La colonne '{aggregate_column}' n'existe pas dans le dataset.")

    df_grouped = df.groupby(aggregate_column)[value_columns].sum().reset_index()
    if value_order is not None:
        df_grouped[aggregate_column] = pd.Categorical(df_grouped[aggregate_column], categories=value_order,
                                                      ordered=True)
        df_grouped = df_grouped.sort_values(aggregate_column)

    return (df_grouped)


def create_admission_graph(df, value, title, label):
    plt.figure(figsize=(10, 6))
    bar_width = 0.4
    x = range(len(df))
    plt.bar(x, df["admissions_urgent"], width=bar_width, label="Admissions Urgentes", alpha=0.7, align='center')
    plt.bar([i + bar_width for i in x], df["admissions_programmed"], width=bar_width, label="Admissions Programmées",
            alpha=0.7, align='center')

    plt.xlabel(label)
    plt.ylabel("Nombre d'Admissions")
    plt.title(title)
    plt.xticks([i + bar_width / 2 for i in x], df[value], rotation=45)
    plt.legend()

    plt.show()

# df = get_aggregated_data('season')
# create_admission_graph(df, "season", "Admissions selon la saison",
#                        "saison")
