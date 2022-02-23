import pandas as pd
import numpy as np
from os import makedirs, getcwd, listdir
from os.path import join, isfile, exists
import threading

NUM_THREAD = 16

# Load dataframes
segStatus_df = pd.read_csv("dataset/temp_segment_status.csv", index_col="_id",
                           parse_dates=["date"])

# Retrieve list of availabel periods
period_status_path = "dataset/period_status"
periods = [f[:-4]
           for f in listdir(period_status_path) if isfile(join(period_status_path, f))]
# does not have "period_20_30" in database
periods = list(set(periods + segStatus_df["period"].unique().tolist()))

min_date = segStatus_df["date"].min()
max_date = segStatus_df["date"].max()

# Create time-dependent data
path_to_data = join(getcwd(), "dataset", "segment_status")
makedirs(path_to_data, exist_ok=True)


def time_to_period(time):
    """
    00:00:00 -> period_0_00
    01:30:00 -> period_1_30
    """
    hour, minute = time.split(":")[:2]

    hour = "_" + str(int(hour))
    minute = "" if minute == "00" else "_" + minute

    return "period" + hour + minute


def los_to_velocity(LOS):
    """
    Map level of service to velocity
    """
    if LOS == "A":
        return 35.0
    elif LOS == "B":
        return 30.0
    elif LOS == "C":
        return 25.0
    elif LOS == "D":
        return 20.0
    elif LOS == "E":
        return 15.0
    elif LOS == "F":
        return 10.0
    return 45.0


def export_dataset(date_chunk):
    for date in date_chunk:
        day = str(date).split(" ")[0]
        makedirs(join(path_to_data, day), exist_ok=True)

        for period in periods:
            if exists(join("dataset/period_status/", period + ".csv")):
                base_status = pd.read_csv(
                    join("dataset/period_status/", period + ".csv"))
                base_status["date"] = pd.to_datetime(day)
                base_status["weekday"] = date.weekday()
                base_status["velocity"] = list(
                    map(los_to_velocity, base_status["LOS"]))

                segStatus_df_filtered = segStatus_df.loc[(
                    segStatus_df["date"] == day) & (segStatus_df["period"] == period)]

                df = pd.merge(left=base_status, right=segStatus_df_filtered, how="outer", on=[
                              "segment_id", "date", "period", "weekday"])
                df["LOS"] = np.where(df["LOS_y"].isnull(),
                                     df["LOS_x"], df["LOS_y"])
                df["velocity"] = np.where(
                    df["velocity_y"].isnull(), df["velocity_x"], df["velocity_y"])
                df = df.drop(
                    ["LOS_x", "LOS_y", "velocity_x", "velocity_y"], axis=1)
                df.index.name = "_id"
                df.to_csv(join(path_to_data, day, period + ".csv"))
            else:
                segStatus_df_filtered = segStatus_df.loc[(
                    segStatus_df["date"] == day) & (segStatus_df["period"] == period)]
                segStatus_df_filtered.index.name = "_id"
                segStatus_df_filtered.to_csv(
                    join(path_to_data, day, period + ".csv"))


date_list = list(pd.date_range(start=min_date, end=max_date))
date_chunk = np.array_split(np.array(date_list), NUM_THREAD)

threads = []
for i in range(NUM_THREAD):
    thread = threading.Thread(target=export_dataset, args=(date_chunk[i],))
    threads.append(thread)

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

print("Done!")
