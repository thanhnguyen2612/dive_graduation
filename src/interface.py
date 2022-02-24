import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timezone, timedelta
import pickle
import time

from src import *

# Create model object
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Net()
model.load_state_dict(torch.load("src/model/model.pt", map_location=device))

# Read segments csv
segment_df = pd.read_csv("dataset/segments.csv")
length_series = segment_df["length"]

# Load encoders
period_encoder = pickle.load(open("src/encoders/period_encoder.pickle", "rb"))
weekday_encoder = pickle.load(open("src/encoders/weekday_encoder.pickle", "rb"))
day_encoder = pickle.load(open("src/encoders/day_encoder.pickle", "rb"))
month_encoder = pickle.load(open("src/encoders/month_encoder.pickle", "rb"))
peak_encoder = pickle.load(open("src/encoders/peak_encoder.pickle", "rb"))
special_day_encoder = pickle.load(open("src/encoders/special_day_encoder.pickle", "rb"))
label_encoder = pickle.load(open("src/encoders/label_encoder.pickle", "rb"))

# Create encoders

label_encoder = LabelEncoder()
label_encoder.fit(["A", "B", "C", "D", "E", "F"])

# Standard scaling features
scaling_feature = ["length", "long_snode", "lat_snode", "long_enode", "lat_enode"]

# One-hot encoding
segment_df, street_type_encoder = one_hot_encoding_segment(segment_df, "street_type", "auto")
segment_df, street_level_encoder = one_hot_encoding_segment(segment_df, "street_level", "auto")

# Drop unneccessary columns
drop_cols = ["s_node_id", "e_node_id", "street_id", "max_velocity", "street_name"]
segment_df = segment_df.drop(columns=drop_cols)


def preprocessing(data):
    """"
    Input:
        data: batch of data (segment_ids, timestamp, return_periods)
    Output:
        X: batch of preprocessed data
    """

    if type(data["timestamp"]) == list:
        df = pd.DataFrame()
        for ts in data["timestamp"]:
            temp = pd.DataFrame.from_dict(
                {
                    "segment_ids": data["segment_ids"],
                    "timestamp": ts
                }
            )
            df = pd.concat([df, temp]).reset_index(drop=True)
    else:
        df = pd.DataFrame.from_dict(data)

    df["timestamp"] = df["timestamp"] * 0.001  # convert ms to s
    df["date"] = df["timestamp"].apply(
        lambda x: datetime.fromtimestamp(x, tz=timezone(timedelta(hours=7))))
    df["weekday"] = df["date"].apply(lambda x: x.weekday())
    df["period"] = df["date"].apply(lambda x: time_to_period(x.hour, x.minute))
    unique_periods = list(pd.unique(df["period"]))
    df = df.drop(columns=["timestamp"])

    df["peak"] = df["period"].apply(infer_peaks)
    df["special_day"] = df["date"].apply(infer_holiday)

    # Extract day and month then drop "date" column
    df['month'] = df['date'].apply(lambda date: date.month)
    df['day'] = df['date'].apply(lambda date: date.day)
    df.drop(["date"], axis=1, inplace=True)

    # Extract hour and minute
    df['period'] = df['period'].apply(period_to_number)

    # One-hot encoding
    df = one_hot_encoding(df, "period", period_encoder)
    df = one_hot_encoding(df, "weekday", weekday_encoder)
    df = one_hot_encoding(df, "day", day_encoder)
    df = one_hot_encoding(df, "month", month_encoder)
    df = one_hot_encoding(df, "peak", peak_encoder)
    df = one_hot_encoding(df, "special_day", special_day_encoder)

    # Merge with segment df
    df = pd.merge(left=df, right=segment_df, left_on="segment_ids",
                  right_on="_id").drop(columns=["segment_ids", "_id"])

    return df.to_numpy().astype("float32"), unique_periods


def inference(data):
    X, unique_period = preprocessing(data)
    X = torch.from_numpy(X).to(device)
    yhat = model(X)
    y_pred = torch.argmax(yhat, dim=1)
    y = label_encoder.inverse_transform(y_pred).tolist()

    if type(data["timestamp"]) == list:
        d = {}
        number_of_segments = len(data["segment_ids"])
        for i in range(len(unique_period)):
            d[unique_period[i]] = {
                "segment_ids": data["segment_ids"], "LOSes": y[i:i+number_of_segments]}
        return d
    return {"segment_ids": data["segment_ids"], "LOSes": y}


def sequence_inference(data):
    curr_time = time.time() * 1000

    times = []
    LOSes = []
    for segment in data["segment_ids"]:
        X, _ = preprocessing(
            {"segment_ids": [segment], "timestamp": curr_time})
        X = torch.from_numpy(X).to(device)
        yhat = model(X)
        y_pred = torch.argmax(yhat, dim=1)
        y = label_encoder.inverse_transform(y_pred).tolist()
        est_velocity = los_to_velocity(y) / 3.6  # to m/s
        est_time = float(length_series.iloc[segment]) * 1000 / est_velocity

        times.append(est_time)
        LOSes.append(y[0])

        curr_time += est_time

    return {"segment_ids": data["segment_ids"], "LOSes": LOSes, "ETAs": times}
