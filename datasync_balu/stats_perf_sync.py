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
match_id='8pm2qt7pp2m2qgnzwrrg8no8a'
path='C:\\Users\\mibam\\egyetem\\sports_analytics\\BEL_data\\test\\'
tracking_fn=f'flat-tracking-{match_id}-25fps.csv'
event_fn=f'events-ma13-with-features-{match_id}.csv'
tracking_df = pd.read_csv(path+tracking_fn)
event_df = pd.read_csv(path+event_fn)

half1_team1_attackdir=event_df.loc[0,'half1_team1_attackdir']

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
    for elem in values:
        i=values.index(elem)
        if elem == playerId:
            x=frame[i+2]
            y=frame[i+3]
    return x,y

#calculate distance metric for an event and trackinf frame pair
def calc_distance(event,frame,justTime):
    if not justTime:
        #calc time distance
        ev_time = event['timeStamp']
        fr_time = frame['timestamp']
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
        ev_time = event['timeStamp']
        fr_time = frame['timestamp']
        dist = distance_time(ev_time,fr_time)
    return (dist)

#evaluates the syncing - with the difference between tracking player position and event position
def eval(event_df,tracking_df):
    distances=[]
    distances_ball=[]
    distances_x=[]
    distances_y=[]
    for ev_i,event in event_df.iterrows():
        frame = tracking_df.iloc[event['frame_id']]
        player_x, player_y = get_player_pos(event['playerTrackingId'],frame)
        distance = distance_pos(event['x'],event['y'],player_x,player_y)
        distances_ball.append(distance_pos(event['x'],event['y'],frame['ball_x'],frame['ball_y']))
        distances_x.append(abs(event['x']-player_x))
        distances_y.append(abs(event['y']-player_y))
        distances.append(distance)
    return distances,distances_ball,distances_x,distances_y

#match events until distance gets lower, stop when it doesn't
def match_events(event_df,tracking_df,justTime):
    frame_ids=[]
    distances=[]
    tr_i=0
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

#look at the next 1500 tracking records and choose the one with the smallest distance
def match_events_2(event_df,tracking_df,justTime):
    frame_ids=[]
    distances=[]
    tr_i=0
    for ev_i, ev_elem in event_df.iterrows():
        fr_dist=[]
        end_i = min(len(tracking_df.index),tr_i+1500)
        for i in range(tr_i,end_i):
            fr_dist.append(calc_distance(ev_elem,tracking_df.iloc[i],justTime))    
        min_dist=min(fr_dist)
        tr_i=tr_i+fr_dist.index(min_dist)
        frame_ids.append(tr_i)
        distances.append(min_dist)
    return frame_ids,distances

def match_events_3(event_df,tracking_df):
    frame_ids=[]
    for ev_i, event in event_df.iterrows():
        rounded_time = int(40* round(float(event['timeStamp'])/40))
        fr_id_l = tracking_df.index[tracking_df['timestamp']==rounded_time].tolist()
        if fr_id_l:
            frame_ids.append(fr_id_l[0])
        else:
            print("error")
    return frame_ids


#frame_ids,distances=match_events(event_df,tracking_df,True)
#frame_ids2,distances2=match_events(event_df,tracking_df,False)
frame_ids3=match_events_3(event_df,tracking_df)
#frame_ids4,distances4=match_events_2(event_df,tracking_df,False)


#event_df2=event_df.copy()
event_df3=event_df.copy()
#event_df4=event_df.copy()


#event_df['frame_id']=frame_ids
#event_df['distance']=distances
#event_df2['frame_id']=frame_ids2
#event_df2['distance']=distances2
event_df3['frame_id']=frame_ids3
#event_df4['frame_id']=frame_ids4
#event_df4['distance']=distances4


#dist,dist_b,dist_x,dist_y=eval(event_df,tracking_df)
#dist2,dist_b2,dist_x2,dist_y2=eval(event_df2,tracking_df)
dist3,dist_b3,dist_x3,dist_y3=eval(event_df3,tracking_df)
event_df3['distance']=dist3
print(event_df3['distance'].max())
event_df3 = event_df3[event_df3['distance']<15]
print(event_df3['distance'].max())

#print(f"Player distance : {dist}, ball distance: {dist_b}, x distance: {dist_x}, y distance: {dist_y}")
#print(f"Player distance : {dist2}, ball distance: {dist_b2}, x distance: {dist_x2}, y distance: {dist_y2}")
print(f"Player distance : {mean(dist3)}, ball distance: {mean(dist_b3)}, x distance: {mean(dist_x3)}, y distance: {mean(dist_y3)}")

#print(eval(event_df3,tracking_df))
#print(eval(event_df4,tracking_df))

#print(event_df.head(1))
#print(tracking_df.iloc[event_df.head(1)['frame_id']])
tracking_df['frame_id'] = tracking_df.index

tracking_w_events =pd.merge(tracking_df,event_df3,how='outer',on=['frame_id'],suffixes=('_tracking','_event'))



tracking_w_events.to_csv(path+f'tracking_w_events_{match_id}.csv',index=False)
synced_events = tracking_w_events[pd.notna(tracking_w_events['eventId'])].copy()

GOAL_X = 100
GOAL_Y = 50
GOALPOST_1Y = 55.3           #was 53.66
GOALPOST_2Y = 44.7           #was 46.34
DIST_GOALPOSTS = 10.6        #a;  was 7.22

def calc_angle_and_distance_to_goal(X,Y, needs_flipping):
    if needs_flipping:
        X = 105-X
    
    diff_X = abs(GOAL_X - X)
    diff_Y = abs(GOAL_Y - Y) 
    dist_to_goal = np.sqrt(diff_X ** 2 + diff_Y ** 2)
    
    diff_gp1y = abs(GOALPOST_1Y - Y)
    diff_gp2y = abs(GOALPOST_2Y - Y)
    dist_gp1y = np.sqrt(diff_X ** 2 + diff_gp1y ** 2) #b
    dist_gp2y = np.sqrt(diff_X ** 2 + diff_gp2y ** 2) #c
    ang_to_goal = np.arccos((dist_gp1y**2 + dist_gp2y**2 - DIST_GOALPOSTS**2)/(2*dist_gp1y*dist_gp2y))
    
    return dist_to_goal, ang_to_goal


for ev_idx, event in synced_events.iterrows():
    ball_carrier = event['playerTrackingId']
    bc_x,bc_y = get_player_pos(ball_carrier,event)
    
    for num in range(1,23):
        p_x,p_y = event[f'player_{num}_x'],event[f'player_{num}_y']
        dist = distance_pos(bc_x,bc_y,p_x,p_y)
        if dist != 0:
            synced_events.loc[ev_idx,f'player_{num}_dist_to_bc'] = dist
            synced_events.loc[ev_idx,f'player_{num}_is_bc'] = 0
        else:
            synced_events.loc[ev_idx,f'player_{num}_dist_to_bc'] = dist
            synced_events.loc[ev_idx,f'player_{num}_is_bc'] = 1
            synced_events.loc[ev_idx,'event_team'] = event[f'player_{num}_teamId']
    
    # ball
    ball_x, ball_y = event['ball_x'],event['ball_y']
    dist_b = distance_pos(bc_x,bc_y,ball_x,ball_y)
    synced_events.loc[ev_idx,'ball_dist_to_bc'] = dist_b
    synced_events.loc[ev_idx,'ball_is_bc'] = 0
    
# ball and player distance- and angle to goal 
second_half_idx = synced_events.half_indicator.idxmax()

if half1_team1_attackdir == 'Left to Right':
     # away_players - calc_angle_and_distance_to_goal(away player X, away player Y, False)
    for num in range(11,21):
        a_p_X_fh = synced_events.loc[:second_half_idx,f'player_{num}_x']
        a_p_Y_fh = synced_events.loc[:second_half_idx,f'player_{num}_y']
        a_p_X_sh = synced_events.loc[second_half_idx:,f'player_{num}_x']
        a_p_Y_sh = synced_events.loc[second_half_idx:,f'player_{num}_y']

        synced_events.loc[:second_half_idx,f'player_{num}_dist_to_goal'],\
            synced_events.loc[:second_half_idx,f'player_{num}_angel_to_goal']=calc_angle_and_distance_to_goal(a_p_X_fh,a_p_Y_fh,True)
        synced_events.loc[second_half_idx:,f'player_{num}_dist_to_goal'],\
                synced_events.loc[second_half_idx:,f'player_{num}_angel_to_goal']=calc_angle_and_distance_to_goal(a_p_X_sh,a_p_Y_sh,False)
    # away goalkeeper
    a_gk_X_fh = synced_events.loc[:second_half_idx,'player_22_x']
    a_gk_Y_fh = synced_events.loc[:second_half_idx,'player_22_y']
    a_gk_X_sh = synced_events.loc[second_half_idx:,'player_22_x']
    a_gk_Y_sh = synced_events.loc[second_half_idx:,'player_22_y']
    synced_events.loc[:second_half_idx,'player_22_dist_to_goal'],\
            synced_events.loc[:second_half_idx,'player_22_angel_to_goal']=calc_angle_and_distance_to_goal(a_gk_X_fh,a_gk_Y_fh,True)
    synced_events.loc[second_half_idx:,'player_22_dist_to_goal'],\
            synced_events.loc[second_half_idx:,'player_22_angel_to_goal']=calc_angle_and_distance_to_goal(a_gk_X_sh,a_gk_Y_sh,False)
    # home players
    for num in range(1,11):
        h_p_X_fh = synced_events.loc[:second_half_idx,f'player_{num}_x']
        h_p_Y_fh = synced_events.loc[:second_half_idx,f'player_{num}_y']
        h_p_X_sh = synced_events.loc[second_half_idx:,f'player_{num}_x']
        h_p_Y_sh = synced_events.loc[second_half_idx:,f'player_{num}_y']

        synced_events.loc[:second_half_idx,f'player_{num}_dist_to_goal'],\
            synced_events.loc[:second_half_idx,f'player_{num}_angel_to_goal']=calc_angle_and_distance_to_goal(h_p_X_fh,h_p_Y_fh,False)
        synced_events.loc[second_half_idx:,f'player_{num}_dist_to_goal'],\
            synced_events.loc[second_half_idx:,f'player_{num}_angel_to_goal']=calc_angle_and_distance_to_goal(h_p_X_sh,h_p_Y_sh,True)
    #hom
    #home goalkeeper - calc_angle_and_distance_to_goal(home player X, home player Y, True)
    h_gk_X_fh = synced_events.loc[:second_half_idx,'player_21_x']
    h_gk_Y_fh = synced_events.loc[:second_half_idx,'player_21_y']
    h_gk_X_sh = synced_events.loc[second_half_idx:,'player_21_x']
    h_gk_Y_sh = synced_events.loc[second_half_idx:,'player_21_y']
    synced_events.loc[:second_half_idx,'player_21_dist_to_goal'],\
            synced_events.loc[:second_half_idx,'player_21_angel_to_goal']=calc_angle_and_distance_to_goal(h_gk_X_fh,h_gk_Y_fh,False)
    synced_events.loc[second_half_idx:,'player_21_dist_to_goal'],\
            synced_events.loc[second_half_idx:,'player_21_angel_to_goal']=calc_angle_and_distance_to_goal(h_gk_X_sh,h_gk_Y_sh,True)

    # ball
    b_X_fh = synced_events.loc[:second_half_idx,'ball_x']
    b_Y_fh = synced_events.loc[:second_half_idx,'ball_y']
    b_X_sh = synced_events.loc[second_half_idx:,'ball_x']
    b_Y_sh = synced_events.loc[second_half_idx:,'ball_y']

    if event['event_team'] == 0:
        synced_events.loc[:second_half_idx,'ball_dist_to_goal'],\
            synced_events.loc[:second_half_idx,'ball_angel_to_goal']=calc_angle_and_distance_to_goal(b_X_fh,b_Y_fh,False)
        synced_events.loc[second_half_idx:,'ball_dist_to_goal'],\
            synced_events.loc[second_half_idx:,'ball_angel_to_goal']=calc_angle_and_distance_to_goal(b_X_sh,b_Y_sh,True)
        #calc_angle_and_distance_to_goal(ball X, ball Y, True)
    else:
        synced_events.loc[:second_half_idx,'ball_dist_to_goal'],\
            synced_events.loc[:second_half_idx,'ball_angel_to_goal']=calc_angle_and_distance_to_goal(b_X_fh,b_Y_fh,True)
        synced_events.loc[second_half_idx:,'ball_dist_to_goal'],\
            synced_events.loc[second_half_idx:,'ball_angel_to_goal']=calc_angle_and_distance_to_goal(b_X_sh,b_Y_sh,False)
        #calc_angle_and_distance_to_goal(ball X, ball Y, False)
    
elif half1_team1_attackdir == 'Right to Left':
     # away_players - calc_angle_and_distance_to_goal(away player X, away player Y, False)
    for num in range(11,21):
        a_p_X_fh = synced_events.loc[:second_half_idx,f'player_{num}_x']
        a_p_Y_fh = synced_events.loc[:second_half_idx,f'player_{num}_y']
        a_p_X_sh = synced_events.loc[second_half_idx:,f'player_{num}_x']
        a_p_Y_sh = synced_events.loc[second_half_idx:,f'player_{num}_y']

        synced_events.loc[:second_half_idx,f'player_{num}_dist_to_goal'],\
            synced_events.loc[:second_half_idx,f'player_{num}_angel_to_goal']=calc_angle_and_distance_to_goal(a_p_X_fh,a_p_Y_fh,False)
        synced_events.loc[second_half_idx:,f'player_{num}_dist_to_goal'],\
                synced_events.loc[second_half_idx:,f'player_{num}_angel_to_goal']=calc_angle_and_distance_to_goal(a_p_X_sh,a_p_Y_sh,True)
    # away goalkeeper
    a_gk_X_fh = synced_events.loc[:second_half_idx,'player_22_x']
    a_gk_Y_fh = synced_events.loc[:second_half_idx,'player_22_y']
    a_gk_X_sh = synced_events.loc[second_half_idx:,'player_22_x']
    a_gk_Y_sh = synced_events.loc[second_half_idx:,'player_22_y']
    synced_events.loc[:second_half_idx,'player_22_dist_to_goal'],\
            synced_events.loc[:second_half_idx,'player_22_angel_to_goal']=calc_angle_and_distance_to_goal(a_gk_X_fh,a_gk_Y_fh,False)
    synced_events.loc[second_half_idx:,'player_22_dist_to_goal'],\
            synced_events.loc[second_half_idx:,'player_22_angel_to_goal']=calc_angle_and_distance_to_goal(a_gk_X_sh,a_gk_Y_sh,True)
    # home players
    for num in range(1,11):
        h_p_X_fh = synced_events.loc[:second_half_idx,f'player_{num}_x']
        h_p_Y_fh = synced_events.loc[:second_half_idx,f'player_{num}_y']
        h_p_X_sh = synced_events.loc[second_half_idx:,f'player_{num}_x']
        h_p_Y_sh = synced_events.loc[second_half_idx:,f'player_{num}_y']

        synced_events.loc[:second_half_idx,f'player_{num}_dist_to_goal'],\
            synced_events.loc[:second_half_idx,f'player_{num}_angel_to_goal']=calc_angle_and_distance_to_goal(h_p_X_fh,h_p_Y_fh,True)
        synced_events.loc[second_half_idx:,f'player_{num}_dist_to_goal'],\
            synced_events.loc[second_half_idx:,f'player_{num}_angel_to_goal']=calc_angle_and_distance_to_goal(h_p_X_sh,h_p_Y_sh,False)
    #home goalkeeper - calc_angle_and_distance_to_goal(home player X, home player Y, True)
    h_gk_X_fh = synced_events.loc[:second_half_idx,'player_21_x']
    h_gk_Y_fh = synced_events.loc[:second_half_idx,'player_21_y']
    h_gk_X_sh = synced_events.loc[second_half_idx:,'player_21_x']
    h_gk_Y_sh = synced_events.loc[second_half_idx:,'player_21_y']
    synced_events.loc[:second_half_idx,'player_21_dist_to_goal'],\
            synced_events.loc[:second_half_idx,'player_21_angel_to_goal']=calc_angle_and_distance_to_goal(h_gk_X_fh,h_gk_Y_fh,True)
    synced_events.loc[second_half_idx:,'player_21_dist_to_goal'],\
            synced_events.loc[second_half_idx:,'player_21_angel_to_goal']=calc_angle_and_distance_to_goal(h_gk_X_sh,h_gk_Y_sh,False)

    # ball
    b_X_fh = synced_events.loc[:second_half_idx,'ball_x']
    b_Y_fh = synced_events.loc[:second_half_idx,'ball_y']
    b_X_sh = synced_events.loc[second_half_idx:,'ball_x']
    b_Y_sh = synced_events.loc[second_half_idx:,'ball_y']

    if event['event_team'] == 0:
        synced_events.loc[:second_half_idx,'ball_dist_to_goal'],\
            synced_events.loc[:second_half_idx,'ball_angel_to_goal']=calc_angle_and_distance_to_goal(b_X_fh,b_Y_fh,True)
        synced_events.loc[second_half_idx:,'ball_dist_to_goal'],\
            synced_events.loc[second_half_idx:,'ball_angel_to_goal']=calc_angle_and_distance_to_goal(b_X_sh,b_Y_sh,False)
        #calc_angle_and_distance_to_goal(ball X, ball Y, True)
    else:
        synced_events.loc[:second_half_idx,'ball_dist_to_goal'],\
            synced_events.loc[:second_half_idx,'ball_angel_to_goal']=calc_angle_and_distance_to_goal(b_X_fh,b_Y_fh,False)
        synced_events.loc[second_half_idx:,'ball_dist_to_goal'],\
            synced_events.loc[second_half_idx:,'ball_angel_to_goal']=calc_angle_and_distance_to_goal(b_X_sh,b_Y_sh,True)
        #calc_angle_and_distance_to_goal(ball X, ball Y, False)
    
synced_events.to_csv(path+f'events_w_tracking_{match_id}.csv',index=False)

#print(tracking_w_events.distance.mean())
#print(mean(distances2))