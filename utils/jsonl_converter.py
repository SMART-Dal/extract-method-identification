import pandas as pd

with open("/home/ip1102/projects/def-tusharma/ip1102/Ref-Res/Research/data/archive/file_0000.jsonl",'r') as f:
    json_objs = list(f)

# print(len(json_objs))
df = pd.DataFrame(columns=['method','label'])
print(df.head())
# for json_obj in json_objs:
