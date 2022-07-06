#!/usr/bin/env python
import json
import pandas as pd
pd.set_option('display.max_columns', None)
from flatten_json import flatten
filename_event='events-ma13-7ky4x0axer75pyu4yskq0ynai.json'
filename_metadata = '2021-08-19-jpl-season-2020-2021-squads.csv'
path='C:\\Users\\mibam\\egyetem\\sports_analytics\\BEL_data\\'

with open(path+filename_event) as f:
    d=json.load(f)

def jsonNormalize(data):
    dic_flattened = (flatten(dd) for dd in data)
    df = pd.DataFrame(dic_flattened)
    return df

df1 = jsonNormalize(d['liveData']['event'])
interesting_periods=[1,2,3,4,5]
interesting_event_ids=[1,2,3,7,8,10,11,12,13,14,15,16,41,50,52,58,61,74]
df1 = df1[df1['periodId'].apply(lambda x: x in interesting_periods)]
df1 = df1[df1['typeId'].apply(lambda x: x in interesting_event_ids)]

id_mapping = pd.read_csv(path+filename_metadata)
id_mapping=id_mapping[['matchName','stats_id']]
id_mapping.columns=['playerName','playerTrackingId']

def get_tr_id(playerNames):
  ids=[]
  for playerName in playerNames:
    id = id_mapping[id_mapping['playerName']==playerName]['playerTrackingId'].values[0]
    ids.append(id)
  return ids
df1['playerTrackingId']=get_tr_id(df1['playerName'].values)

def getmillisecs(timestamps):
  out=[]
  for timestamp in timestamps:
    #print(timestamp)
    time = str(timestamp).split('T')[1][:-1]
    timelist=time.split(':')
    temp=timelist[2].split('.')
    timelist[2]=temp[0]
    if len(temp)>1:
      timelist.append(temp[1])
      out.append(int(timelist[3])+int(timelist[2])*1000+int(timelist[1])*60*1000+int(timelist[0])*60*60*1000)
    else:
      out.append(int(timelist[2])*1000+int(timelist[1])*60*1000+int(timelist[0])*60*60*1000)
  return out

test1 = getmillisecs(df1[df1['periodId']==1]['timeStamp'].values.tolist())
test2 = getmillisecs(df1[df1['periodId']==2]['timeStamp'].values.tolist())
test1 = pd.DataFrame(test1,columns=['timeStamp'])
test1['1']=test1['timeStamp']-test1['timeStamp'].values[0]
test1['1']=test1['1'].fillna(0)
test2 = pd.DataFrame(test2,columns=['timeStamp'])
test2['1']=test2['timeStamp']-test2['timeStamp'].values[0]
test2['1']=test2['1'].fillna(0)
test=pd.concat([test1,test2])

df1['time_of_current_half']=test['1'].values.tolist()
temp = df1[df1['periodId']==1]['time_of_current_half'].max()
df1['timeMilliSec'] = df1['time_of_current_half']+temp*(df1['periodId']-1)


df1.to_csv(path+'events-ma13-7ky4x0axer75pyu4yskq0ynai.csv',index=False)
print(df1)