from pymongo import MongoClient
import pandas as pd

uri = "mongodb://localhost:27017/bktraffic"
db = MongoClient(uri)["bktraffic"]

basicTrafficStatus = db["Basic_Traffic_Status"]

segment_dict = dict()

for x in basicTrafficStatus.find():
    segment_dict[x["segmentId"]] = x["segmentStatus"]

df = pd.DataFrame(segment_dict)
df = df.drop(labels="sdfsdf", axis=0)
df.index.name = "period"

df.to_csv("dataset/temp_base_status.csv")
