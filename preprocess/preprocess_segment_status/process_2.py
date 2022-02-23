"""
Transform Segment Reports' velocity into dates, weekday, p_names
"""

import numpy as np
import pandas as pd
from math import ceil

segStatus_df = pd.read_csv("dataset/segment_reports.csv", index_col="_id",
                           parse_dates=["updated_at"])
segment_df = pd.read_csv("dataset/temp_segments.csv", index_col="_id",
                         parse_dates=["created_at", "updated_at"])


def transform_LOS(segment_id, velocity):
    max_velocity = segment_df.loc[segment_id, "max_velocity"]
    if max_velocity is None:
        max_velocity = 50

    # Transform to label
    labels = ["A", "B", "C", "D", "E", "F"]
    threshold = 35
    if max_velocity >= 70:
        threshold = 45
    elif max_velocity >= 60:
        threshold = 40

    t = max(threshold - velocity, 0)
    return labels[min(ceil(t / 5), 5)]


def major_voting(labels):
    unique_labels = set(labels)
    count_labels = [labels.count(label) for label in unique_labels]

    sorted_labels = sorted(
        zip(unique_labels, count_labels), key=lambda x: x[1])
    if len(sorted_labels) > 1 and sorted_labels[0][1] == sorted_labels[1][1]:
        print("Hey, error report!")
    return sorted_labels[0][0]


def mean_voting(labels):
    return np.mean(labels)


def transform_report(row):
    """
    @Params:
        dt: Timestamp object of Pandas
    @Return:
        dict: {"date", "period_{hour}_{00|30}"}
    """
    dt = row["updated_at"]
    h = dt.hour
    m = "00" if dt.minute < 30 else "30"
    p_name = f"period_{h}_{m}"
    return dt.date(), dt.weekday(), p_name


dates = []
weekdays = []
p_names = []

for _, row in segStatus_df.iterrows():
    date, weekday, p_name = transform_report(row)
    dates.append(date)
    weekdays.append(weekday)
    p_names.append(p_name)

segStatus_df["date"] = dates
segStatus_df["weekday"] = weekdays
segStatus_df["period"] = p_names

segStatus_df = segStatus_df.groupby(
    by=["segment_id", "date", "weekday", "period"])["velocity"].apply(list)
segStatus_df = pd.DataFrame(segStatus_df).reset_index()
segStatus_df["velocity"] = segStatus_df["velocity"].apply(mean_voting)

LOSes = []
for _, row in segStatus_df.iterrows():
    LOS = transform_LOS(row["segment_id"], row["velocity"])
    LOSes.append(LOS)

segStatus_df["LOS"] = LOSes

segStatus_df.to_csv("dataset/temp_segment_status.csv", index_label="_id")
