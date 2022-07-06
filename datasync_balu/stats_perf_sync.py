#need:  -function(distance_pos) that calcs distance between two points
#       -function(distance_time) that calcscthe "distance" between the tracking and event record's timestamp 
#           (one is in milliseconds, other is minutes and seconds separately, so rounding is necessary) 
#       -function that loops through the eventlist it recieves and does what's described in the how section
#            with the help of the distance_pos and distance_time function



#what I want:   loop through the events of a match and pair them with tracking data from the same match.
#
#how:           take first event record, measure distance between event point (event_x,event_y) and ball's position
#               , then event point to player's position (who 'commits' the event)
#               Do this, while the distances are getting lower. If distances are increasing stop and match the event
#               record with the last non-ncreasing tracking record.
#               Do this for every event record.
# 
#               #2 -- Try with event data timeStamp??-has miliseconds! 

#TRACKING DATA NEEDS TO BE CLEANED FIRST - done
from inspect import FrameInfo
from math import sqrt
from statistics import mean
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

#opening files
path='C:\\Users\\mibam\\egyetem\\sports_analytics\\BEL_data\\'
tracking_fn='flat-tracking-7ky4x0axer75pyu4yskq0ynai-25fps.csv'
event_fn='events-ma13-with-features-7ky4x0axer75pyu4yskq0ynai.csv'
tracking_df = pd.read_csv(path+tracking_fn)
event_df = pd.read_csv(path+event_fn)
#print(tracking_df.head(1))
#print(event_df.sample(1))

#helper functions for calculating disance between timestamps
def distance_time(time_from, time_to):
    return abs(time_to - time_from)
def conv_milisec_to_sec(milliseconds):
    return round((milliseconds)/1000)
def conv_min_sec_to_sec(minute,second):
    return (minute*60+second)
#helper functions for calculating distance between positions
def distance_pos(x_1,y_1,x_2,y_2):
    return sqrt((x_2-x_1)**2+(y_2-y_1)**2)


def get_player_pos(playerId,frame):
    x=0
    y=0
    values =frame.values.tolist()
    #print(values)
    for elem in values:
        i=values.index(elem)
        #print(f"Index : {i}, Value : {elem}")
        if elem == playerId:
            x=frame[i+2]
            y=frame[i+3]
    #print(playerId)
    #print('({};{})'.format(x,y))
    return x,y

#calculate distance metric for an event and trackinf frame pair
def calc_distance(event,frame,justTime):
    if not justTime:
        #calc time distance
        ev_time = event['timeMilliSec']
        fr_time = frame['timestamp']
        #ev_time = conv_min_sec_to_sec(event['timeMin'],event['timeSec'])
        #fr_time = conv_milisec_to_sec(frame['time_of_current_half'])
        time_dist = distance_time(ev_time,fr_time)
        #calc positional distance
        #   -ball
        pos_distance_ball = distance_pos(event['x'],event['y'],frame['ball_x'],frame['ball_y'])
        #   -player performing the event
        player = event['playerTrackingId']
        player_x,player_y = get_player_pos(player,frame)
        pos_distance_player = distance_pos(event['x'],event['y'],player_x,player_y)
        #return overall distance
        dist=time_dist*0.33+pos_distance_ball*0.33+pos_distance_player*0.33
    else:
        #calc time distance
        ev_time = event['timeMilliSec']
        fr_time = frame['timestamp']
        dist = distance_time(ev_time,fr_time)
    #print(dist)
    return (dist)

#evaluates the syncing - with the difference between tracking player position and event position
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
    return mean(distances)

def match_events(event_df,tracking_df,justTime):
    frame_ids=[]
    distances=[]
    tr_i=0
    #ev_i=0
    for ev_i, ev_elem in event_df.iterrows():
        min_dist=calc_distance(ev_elem,tracking_df.iloc[tr_i],justTime)
        dist = calc_distance(ev_elem,tracking_df.iloc[tr_i+1],justTime)
        while (dist<=min_dist):
            tr_i+=1
            min_dist=calc_distance(ev_elem,tracking_df.iloc[tr_i],justTime)
            dist = calc_distance(ev_elem,tracking_df.iloc[tr_i+1],justTime)
        frame_ids.append(tr_i)
        distances.append(min_dist)
    return frame_ids,distances

def match_events_2(event_df,tracking_df,justTime):
    frame_ids=[]
    distances=[]
    tr_i=0
    #ev_i=0
    for ev_i, ev_elem in event_df.iterrows():
        fr_dist=[]
        end_i = min(len(tracking_df.index),tr_i+1500)
        for i in range(tr_i,end_i):
            fr_dist.append(calc_distance(ev_elem,tracking_df.iloc[i],justTime))    
            #print(f"Index: {i}")   
        min_dist=min(fr_dist)
        tr_i=tr_i+fr_dist.index(min_dist)
        frame_ids.append(tr_i)
        distances.append(min_dist)
    return frame_ids,distances

#frame_ids,distances=match_events(event_df,tracking_df,True)
frame_ids2,distances2=match_events_2(event_df,tracking_df,True)
#frame_ids3,distances3=match_events(event_df,tracking_df,False)
#frame_ids4,distances4=match_events_2(event_df,tracking_df,False)


event_df2=event_df.copy()
#event_df3=event_df.copy()
#event_df4=event_df.copy()


#event_df['frame_id']=frame_ids
#event_df['distance']=distances
event_df2['frame_id']=frame_ids2
event_df2['distance']=distances2
#event_df3['frame_id']=frame_ids3
#event_df3['distance']=distances3
#event_df4['frame_id']=frame_ids4
#event_df4['distance']=distances4



#print(eval(event_df,tracking_df))
print(eval(event_df2,tracking_df))
#print(eval(event_df3,tracking_df))
#print(eval(event_df4,tracking_df))

#print(event_df.head(1))
#print(tracking_df.iloc[event_df.head(1)['frame_id']])
tracking_df['frame_id'] = tracking_df.index

tracking_w_events =pd.merge(tracking_df,event_df2,how='outer',on=['frame_id'],suffixes=('_tracking','_event'))
tracking_w_events.to_csv(path+'tracking_w_events_test6.csv')
#print(tracking_w_events.distance.mean())
#print(mean(distances2))