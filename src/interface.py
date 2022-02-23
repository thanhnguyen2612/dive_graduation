import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timezone, timedelta
import pickle
import time

from src.utils import *
from src.model import *

# Create model object
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Net()
model.load_state_dict(torch.load(
    "src/model/model_best_epoch_0.pt", map_location=device))

# Read segments csv
segment_df = pd.read_csv("dataset/segments.csv")
length_series = segment_df["length"]

# Load encoders
period_encoder = pickle.load(open("src/encoders/period_encoder.pickle", "rb"))
weekday_encoder = pickle.load(
    open("src/encoders/weekday_encoder.pickle", "rb"))
day_encoder = pickle.load(open("src/encoders/day_encoder.pickle", "rb"))
month_encoder = pickle.load(open("src/encoders/month_encoder.pickle", "rb"))
peak_encoder = pickle.load(open("src/encoders/peak_encoder.pickle", "rb"))
special_day_encoder = pickle.load(
    open("src/encoders/special_day_encoder.pickle", "rb"))
label_encoder = pickle.load(open("src/encoders/label_encoder.pickle", "rb"))

# Create encoders

label_encoder = LabelEncoder()
label_encoder.fit(["A", "B", "C", "D", "E", "F"])

# Standard scaling features
scaling_feature = ["length", "long_snode",
                   "lat_snode", "long_enode", "lat_enode"]

# One-hot encoding
segment_df, street_type_encoder = one_hot_encoding_segment(
    segment_df, "street_type", "auto")
segment_df, street_level_encoder = one_hot_encoding_segment(
    segment_df, "street_level", "auto")

# Drop unneccessary columns
segment_df = segment_df.drop(
    columns=["s_node_id", "e_node_id", "street_id", "max_velocity", "street_name"])


def time_preprocessing(data, timestamp=None):
    """"
    Input:
        data: batch of data (time, segments_id)
    Output:
        X: batch of preprocessed data
    """

    df = pd.DataFrame.from_dict(data)

    if timestamp is not None:
        df["time"] = timestamp
    df["date"] = df["time"].apply(
        lambda x: datetime.fromtimestamp(x, tz=timezone(timedelta(hours=7))))
    df["weekday"] = df["date"].apply(lambda x: x.weekday())
    df["period"] = df["date"].apply(lambda x: time_to_period(x.hour, x.minute))
    df = df.drop(columns=["time"])

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
    df = pd.merge(left=df, right=segment_df, left_on="segment_id",
                  right_on="_id").drop(columns=["segment_id", "_id"])

    return df.to_numpy().astype("float32")


def period_preprocessing(data):
    """"
    Input:
        data: Dict(segment_id: List, date: List, period: List)
    Output:
        X: batch of preprocessed data
    """

    df = pd.DataFrame.from_dict(data)

    df["date"] = df["date"].apply(
        lambda x: datetime.strptime(x, "%a %b %d %Y"))
    df["weekday"] = df["date"].apply(lambda x: x.weekday())

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
    df = pd.merge(left=df, right=segment_df, left_on="segment_id",
                  right_on="_id").drop(columns=["segment_id", "_id"])

    return df.to_numpy().astype("float32")


def time_inference(data):
    X = time_preprocessing(data)
    X = torch.from_numpy(X).to(device)
    yhat = model(X)
    y_pred = torch.argmax(yhat, dim=1)
    y = label_encoder.inverse_transform(y_pred).tolist()

    return {"segment_ids": data["segment_id"], "LOSes": y}


def period_inference(data):
    X = period_preprocessing(data)
    X = torch.from_numpy(X).to(device)
    yhat = model(X)
    y_pred = torch.argmax(yhat, dim=1)
    y = label_encoder.inverse_transform(y_pred).tolist()

    return {"segment_id": data["segment_id"], "LOS": y}


def sequence_inference(data):
    curr_time = time.time()

    times = []
    LOSes = []
    for segment in data["segment_id"]:
        X = time_preprocessing({"segment_id": [segment]}, curr_time)
        X = torch.from_numpy(X).to(device)
        yhat = model(X)
        y_pred = torch.argmax(yhat, dim=1)
        y = label_encoder.inverse_transform(y_pred).tolist()
        est_velocity = los_to_velocity(y) / 3.6  # to m/s
        est_time = float(length_series.iloc[segment]) / est_velocity

        times.append(est_time)
        LOSes.append(y)

        curr_time += est_time

    return {"segment_id": data["segment_id"], "LOS": LOSes, "ETA": times}
