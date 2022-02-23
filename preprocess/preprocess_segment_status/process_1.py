"""
    Label encoding indexes of segment reports
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data_origin/segment_reports.csv")

le = LabelEncoder()
le.fit(df["_id"])

df["_id"] = le.transform(df["_id"])

cols = ["_id", "updatedAt", "segment", "velocity"]
new_df = df[cols].rename(
    columns={"updatedAt": "updated_at", "segment": "segment_id"})

new_df.to_csv("dataset/segment_reports.csv", index=False)
