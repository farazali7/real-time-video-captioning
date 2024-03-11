import pandas as pd
import json


path = 'data/MSRVTT/annotation/MSR_VTT.json'
with open(path) as json_data:
    data = json.load(json_data)
    df = pd.DataFrame(data['annotations'])
    df.to_csv('data/MSRVTT/labels.csv')

print('Done')

