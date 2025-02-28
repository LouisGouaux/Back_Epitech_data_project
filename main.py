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


def get_admissions_data():
    df = import_dataset()
    df_weekly = df.groupby("day_of_week")[["admissions_urgent", "admissions_programmed"]].sum().reset_index()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df_weekly["day_of_week"] = pd.Categorical(df_weekly["day_of_week"], categories=order, ordered=True)
    df_weekly = df_weekly.sort_values("day_of_week")
    return(df_weekly)

def create_admission_graph(df, value, title, label):
        plt.figure(figsize=(10, 6))
        bar_width = 0.4
        x = range(len(df))
        plt.bar(x, df["admissions_urgent"], width=bar_width, label="Admissions Urgentes", alpha=0.7, align='center')
        plt.bar([i + bar_width for i in x], df["admissions_programmed"], width=bar_width, label="Admissions Programmées", alpha=0.7, align='center')

        plt.xlabel(label)
        plt.ylabel("Nombre d'Admissions")
        plt.title(title)
        plt.xticks([i + bar_width / 2 for i in x], df["day_of_week"], rotation=45)
        plt.legend()

        plt.show()


create_admission_graph(get_admissions_data(), "day_of_week", "Admissions selon le jour de la semaine", "Jour de la semaine")