import pandas as pd

def import_dataset():
    df = pd.read_csv("./dataset/hospital_data.csv")
    return(df)

def get_resources(date):
    df = import_dataset()
    df_filtered = df[df["date"] == date]
    resource_column = [
        "service", "beds_occupied", "beds_available", "staff_present"
    ]
    return(df_filtered[["date"] + resource_column].to_dict(orient="records"))

