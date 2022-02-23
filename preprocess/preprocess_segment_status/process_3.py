"""
Create metadata
"""

import pandas as pd

# Load dataframes
segment_df = pd.read_csv("dataset/temp_segments.csv",
                         parse_dates=["created_at", "updated_at"])
node_df = pd.read_csv("dataset/nodes.csv", index_col='_id')

# Create metadata of segment
df = pd.merge(left=segment_df.drop(columns=["created_at", "updated_at"]), right=node_df, how="left",
              left_on="s_node_id", right_on="_id")

df = pd.merge(left=df, right=node_df, how="left",
              left_on="e_node_id", right_on="_id", suffixes=("_snode", "_enode"))

df.set_index("_id").to_csv("dataset/segments.csv", index_label="_id")
