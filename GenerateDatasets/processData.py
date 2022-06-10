import pandas as pd
import numpy as np
import os
import glob

def processWeather(weather_path = "../Datasets/Weather/"):
    files = glob.glob(weather_path+"*.csv")
    dfs = []
    for file_path in files:
        df = pd.read_csv(file_path)
        df['Day'] = pd.DatetimeIndex(df['datetime']).day
        df['Month'] = pd.DatetimeIndex(df['datetime']).month
        df['Year'] = pd.DatetimeIndex(df['datetime']).year
        df['Name'].replace({'Shanghai,China': 'Shanghai'}, inplace=True)
        df.rename(columns={'name': 'Site'}, inplace=True)
        dfs.append(df[["Site", 'Year', 'Month', 'Day', "temp", "humidity", "precip", "sealevelpressure", "windspeed", "winddir", "cloudcover", "uvindex"]].copy())
    df_data = pd.concat(dfs)
    df_data = df_data.dropna()
    if not os.path.exists(weather_path+'finalData'):
        os.makedirs(weather_path+'finalData')
    df_data.to_csv(weather_path + "finalData/weather_data.csv", index=False)


def processGT(gt_path):
    gt_files = glob.glob(gt_path+"*.csv")
    dfs = []
    for file_path in gt_files:
        df = pd.read_csv(file_path)
        # get these columns as long as the row is marked as Valid
        dfs.append(
            df.loc[(df['QC Name'] == "Valid"), ['Site', 'Year', 'Month', 'Day', 'Hour', 'AQI Category', 'Raw Conc.']])
    # combine all dataframes for each location
    df_data = pd.concat(dfs)
    df_data = df_data.dropna()
    if not os.path.exists(gt_path + 'finalData'):
        os.makedirs(gt_path + 'finalData')
    df_data.to_csv(gt_path + "finalData/gt_data.csv", index=False)


if __name__ == "__main__":
    weather_path = "../Datasets/Weather/"
    gt_path = "../Datasets/GT/"

    processWeather(weather_path)
    processGT(gt_path)