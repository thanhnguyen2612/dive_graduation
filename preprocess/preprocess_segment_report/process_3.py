import pandas as pd

segStatus_df = pd.read_csv("dataset/temp_segment_reports.csv", index_col="_id",
                           parse_dates=["updated_at", "date"])
segment_df = pd.read_csv("dataset/segments.csv", index_col="_id",
                         parse_dates=["created_at", "updated_at"])
node_df = pd.read_csv("dataset/nodes.csv", index_col='_id')


def major_voting(labels):
    unique_labels = set(labels)
    count_labels = [labels.count(label) for label in unique_labels]

    sorted_labels = sorted(
        zip(unique_labels, count_labels), key=lambda x: x[1])
    if len(sorted_labels) > 1 and sorted_labels[0][1] == sorted_labels[1][1]:
        print("Hey, error report!")
    return sorted_labels[0][0]


def mean_voting(labels):
    l = ["A", "B", "C", "D", "E", "F"]
    values = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
    mean = sum(values[label] for label in labels) / len(labels)
    return l[min(round(mean), 5)]


compact_LOS = segStatus_df.groupby(
    by=["segment_id", "date", "weekday", "period"])["LOS"].apply(list)
compact_LOS = pd.DataFrame(compact_LOS).reset_index()
compact_LOS["LOS"] = compact_LOS["LOS"].apply(mean_voting)

df = pd.merge(left=compact_LOS, right=segment_df.drop(columns=["created_at", "updated_at"]),
              left_on="segment_id", right_on="_id")
df = pd.merge(left=df, right=node_df, how="left",
              left_on="s_node_id", right_on="_id")
df = pd.merge(left=df, right=node_df, how="left",
              left_on="e_node_id", right_on="_id", suffixes=("_snode", "_enode"))

df.to_csv("dataset/test.csv", index_label="_id")
