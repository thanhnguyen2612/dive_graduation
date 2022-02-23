import numpy as np
import pandas as pd
from math import ceil

segReport_df = pd.read_csv("dataset/segment_reports.csv", index_col="_id",
                           parse_dates=["updated_at"])
segment_df = pd.read_csv("dataset/segments.csv", index_col="_id",
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


def transform_report(row):
    """
    @Params:
        dt: Timestamp object of Pandas
    @Return:
        dict: {"date", "period_{hour}_{00|30}"}
    """
    LOS = transform_LOS(row["segment_id"], row["velocity"])
    dt = row["updated_at"]
    intervals = list(range(24))
    h = dt.hour
    m = "00" if dt.minute < 30 else "30"
    p_name = f"period_{h}_{m}"
    return dt.date(), dt.weekday(), p_name, LOS


dates = []
weekdays = []
p_names = []
LOSes = []

for _, row in segReport_df.iterrows():
    date, weekday, p_name, LOS = transform_report(row)
    dates.append(date)
    weekdays.append(weekday)
    p_names.append(p_name)
    LOSes.append(LOS)

segReport_df["date"] = dates
segReport_df["weekday"] = weekdays
segReport_df["period"] = p_names
segReport_df["LOS"] = LOSes

segReport_df.to_csv("dataset/temp_segment_reports.csv", index_label="_id")
