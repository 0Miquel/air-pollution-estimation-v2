import pandas as pd
import requests
import datetime
import glob
import os

from keys import keys
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--location', default='Shanghai')
args = parser.parse_args(sys.argv[1:])

location = args.location


for key in keys:
    while True:
        #get last date recorded and add 1 day to get the new data
        date = glob.glob(f"../Datasets/Weather/scheduled/{location}*.csv")[0].split("_")[-1].split(".")[0]
        datetime_date = datetime.datetime.strptime(date, "%Y-%m-%d")
        next_date = datetime_date + datetime.timedelta(1)
        string_next_date = next_date.strftime("%Y-%m-%d")

        response = requests.get(f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{string_next_date}/{string_next_date}?unitGroup=metric&include=hours&key={key}&contentType=csv")

        if response.status_code == 200:
            url_content = response.content
            csv_file = open('../Datasets/Weather/scheduled/downloaded.csv', 'wb')
            csv_file.write(url_content)
            csv_file.close()

            downloaded_df = pd.read_csv('../Datasets/Weather/scheduled/downloaded.csv')
            downloaded_df['Hour'] = pd.DatetimeIndex(downloaded_df['datetime']).hour
            downloaded_df['Day'] = pd.DatetimeIndex(downloaded_df['datetime']).day
            downloaded_df['Month'] = pd.DatetimeIndex(downloaded_df['datetime']).month
            downloaded_df['Year'] = pd.DatetimeIndex(downloaded_df['datetime']).year
            downloaded_df['name'].replace({'中国上海': 'Shanghai', '中国北京': 'Beijing'}, inplace=True)
            downloaded_df.rename(columns={'name': 'Site'}, inplace=True)
            downloaded_df = downloaded_df[["Site", 'Year', 'Month', 'Day', 'Hour', "temp", "humidity", "precip", "sealevelpressure", "windspeed",
                "winddir", "cloudcover", "uvindex"]].copy()

            data_file = pd.read_csv(f'../Datasets/Weather/scheduled/{location}_{date}.csv')

            df_data = pd.concat([data_file, downloaded_df])
            df_data.to_csv(f"../Datasets/Weather/scheduled/{location}_{string_next_date}.csv", index=False)
            os.remove(f'../Datasets/Weather/scheduled/{location}_{date}.csv')
        else:
            break