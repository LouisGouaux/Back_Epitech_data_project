import pandas as pd
import matplotlib.pyplot as plt


def import_dataset():
    df = pd.read_csv("./dataset/hospital_data.csv")
    return (df)


def get_resources(date):
    df = import_dataset()
    df_filtered = df[df["date"] == date]
    resource_column = [
        "service", "beds_occupied", "beds_available", "staff_present"
    ]
    return (df_filtered[["date"] + resource_column].to_dict(orient="records"))


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
