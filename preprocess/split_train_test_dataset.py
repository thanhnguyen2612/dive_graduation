import pandas as pd

seg_df = pd.read_csv("dataset/segments.csv", index_col="_id")
status_df = pd.read_csv("temp/segment_status_1.csv", index_col="_id")

drop_seg_df = seg_df.drop(columns=["created_at", "updated_at"])

join_df = pd.merge(left=status_df, right=drop_seg_df,
                   left_on="segment_id", right_on="_id")

join_df.to_csv("dataset/daily_traffic_los.csv", index=False)
