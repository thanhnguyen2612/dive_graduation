import pandas as pd
from os import makedirs
from os.path import join

baseStatus_df = pd.read_csv("dataset/temp_base_status.csv")

period_status_path = "dataset/period_status"
makedirs(period_status_path, exist_ok=True)

segment_id = list(baseStatus_df.keys())[1:]


def rename_period(period):
    """
    period_0 -> period_0_00
    period_1_30 -> period_1_30
    """
    suffix = "_00" if len(period.split("_")) == 2 else ""
    return period + suffix


for _, row in baseStatus_df.iterrows():
    period_renamed = rename_period(row["period"])
    period = [period_renamed] * len(segment_id)
    LOS = row.tolist()[1:]
    df = pd.DataFrame(list(zip(period, segment_id, LOS)),
                      columns=["period", "segment_id", "LOS"])
    df.to_csv(join(period_status_path, period_renamed+".csv"), index=False)
