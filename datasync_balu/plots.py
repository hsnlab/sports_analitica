#!/usr/bin/env python
# coding: utf-8

import pandas as pd
pd.set_option('display.max_columns', None)
from math import sqrt
import matplotlib.pyplot as plt

match_id='7ky4x0axer75pyu4yskq0ynai'
path='C:\\Users\\mibam\\egyetem\\sports_analytics\\BEL_data\\'
df = pd.read_csv(path+f'tracking_w_events_{match_id}.csv')

#df of just events (from synced data, so has tracking and event data, and event data is never null)
event_df = df[pd.notna(df['eventId'])]

max_dist=event_df['distance'].max()
print(event_df[event_df['distance']==max_dist])
#event_df['distance_x']=dist_x
#event_df['distance_y']=dist_y

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
print(temp_df['0'].max())
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

#rolling mean plot by teams - axis' are the same as above
#creating subdataframe
contestants=pd.DataFrame(event_df[['contestantId','distance','distance_x','distance_y']].values,columns=['contestantId','distance','distance_x','distance_y'], index=pd.to_datetime(event_df['timestamp'],unit='ms'))
#getting contestantId values
keys = list(contestants.groupby(by='contestantId').groups.keys())
#separate dataframes for the two teams
home_d=contestants[contestants['contestantId']==keys[0]]['distance']
away_d=contestants[contestants['contestantId']==keys[1]]['distance']
home_d_x=contestants[contestants['contestantId']==keys[0]]['distance_x']
away_d_x=contestants[contestants['contestantId']==keys[1]]['distance_x']
home_d_y=contestants[contestants['contestantId']==keys[0]]['distance_y']
away_d_y=contestants[contestants['contestantId']==keys[1]]['distance_y']
#1 minute rolling means for each team
rolling_mean_h_d=home_d.rolling(window='1T').mean()
rolling_mean_a_d=away_d.rolling(window='1T').mean()
rolling_mean_h_d_x=home_d_x.rolling(window='1T').mean()
rolling_mean_a_d_x=away_d_x.rolling(window='1T').mean()
rolling_mean_h_d_y=home_d_y.rolling(window='1T').mean()
rolling_mean_a_d_y=away_d_y.rolling(window='1T').mean()

#team #1
plt.figure(figsize=(20, 10))
plt.plot(rolling_mean_h_d, 'r-', label=f'1 minute running average, {keys[0]}')
plt.plot(rolling_mean_h_d_x, 'b-', label=f'1 minute running average, {keys[0]} x')
plt.plot(rolling_mean_h_d_y, 'g-', label=f'1 minute running average, {keys[0]} y')
plt.ylabel('error')
plt.xlabel('event number')
plt.grid(linestyle=':')
plt.legend(loc='upper left')
plt.show()


#team #2
plt.figure(figsize=(20, 10))
plt.plot(rolling_mean_a_d, 'r-', label=f'1 minute running average, {keys[1]}')
plt.plot(rolling_mean_a_d_x, 'b-', label=f'1 minute running average, {keys[1]} x')
plt.plot(rolling_mean_a_d_y, 'g-', label=f'1 minute running average, {keys[1]} y')
plt.ylabel('error')
plt.xlabel('event number')
plt.grid(linestyle=':')
plt.legend(loc='upper left')
plt.show()

#scatter plot    ---    x axis is the event typeId (what kind of event is it)
#                       y axis is distance
plt.scatter(event_df['typeId'].values,event_df['distance'].values)
plt.show()
