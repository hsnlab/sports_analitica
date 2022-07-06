#!/usr/bin/env python
# coding: utf-8

import pandas as pd
pd.set_option('display.max_columns', None)
from math import sqrt
import matplotlib.pyplot as plt


df = pd.read_csv('C:\\Users\\mibam\\egyetem\\sports_analytics\\BEL_data\\tracking_w_events_test6.csv')

#df of just events (from synced data, so has tracking and event data, and event data is never null)
event_df = df[pd.notna(df['eventId'])]

#helper functions for calculating distance  --- distance is:    difference between the recorded coordinates of the event 
#                                                               and the tracking position of the player who committed the event
#                                                               (data only has events, where playerId is not null, so there's always a player committing the event)
def distance_pos(x_1,y_1,x_2,y_2):
    return sqrt((x_2-x_1)**2+(y_2-y_1)**2)
def get_player_pos(playerId,frame):
    x=0
    y=0
    values =frame.values.tolist()
    for elem in values:
        i=values.index(elem)
        if elem == playerId:
            x=frame[i+2]
            y=frame[i+3]
    return x,y

def eval(event_df,tracking_df):
    distances=[]
    for ev_i,event in event_df.iterrows():
        frame = tracking_df.iloc[event['frame_id']]
        #print(frame)
        player_x, player_y = get_player_pos(event['playerTrackingId'],frame)
        distance = distance_pos(event['x'],event['y'],player_x,player_y)
        #distance = distance_pos(event['x'],event['y'],frame['ball_x'],frame['ball_y'])
        #print(distance)
        distances.append(distance)
    return distances

distances=eval(event_df,df)
event_df['distance']=distances


#cdf and pdf plot of the errors (= distances)
#       x axis is the distance
#       right y axis is the number (index) of event
#       left y axis is frequency
fig, ax = plt.subplots()
ax2 = ax.twinx()
n, bins, patches = ax.hist(event_df['distance'], bins=100)#, normed=False)
n, bins, patches = ax2.hist(
    event_df['distance'], cumulative=1, histtype='step', bins=100, color='tab:orange')


#helper df for the rolling mean plots, with datetime index
temp_df = pd.DataFrame(event_df['distance'].values,columns=['0'], index=pd.to_datetime(event_df['timestamp'],unit='ms'))
#rolling mean calculation
rolling_mean=temp_df.rolling(window='1T').mean()

#rolling mean plot   ---    x axis is number (index) of events
#                           y axis is the rolling mean of the distances, with a one minute window
plt.figure(figsize=(20, 10))
#plt.plot(temp_df['0'].values, 'k-', label='Original')
plt.plot(rolling_mean, 'r-', label='1 minute running average')
plt.ylabel('error')
plt.xlabel('time')
plt.grid(linestyle=':')
plt.legend(loc='upper left')
plt.show()


#scatter plot    ---    x axis is the event typeId (what kind of event is it)
#                       y axis is distance
plt.scatter(event_df['typeId'].values,event_df['distance'].values)
plt.show()
