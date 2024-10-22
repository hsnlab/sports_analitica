#!/usr/bin/env python
import json
import pandas as pd
import numpy as np

#pd.set_option('display.max_columns', None)
from flatten_json import flatten
from dateutil import parser
from math import sqrt,pow

match_id = '7ky4x0axer75pyu4yskq0ynai'
filename_event = f'events-ma13-{match_id}.json'
filename_metadata = '2021-08-19-jpl-season-2020-2021-squads.csv'
path = 'C:\\Users\\mibam\\egyetem\\sports_analytics\\BEL_data\\test\\'

tracking_cols = ['timestamp', 'time_of_current_half', 'half_indicator', 'match_not_in_play',
                 'player_1_teamId', 'player_1_objectId', 'player_1_jerseyNum', 'player_1_x', 'player_1_y',
                 'player_2_teamId', 'player_2_objectId', 'player_2_jerseyNum', 'player_2_x', 'player_2_y',
                 'player_3_teamId', 'player_3_objectId', 'player_3_jerseyNum', 'player_3_x', 'player_3_y',
                 'player_4_teamId', 'player_4_objectId', 'player_4_jerseyNum', 'player_4_x', 'player_4_y',
                 'player_5_teamId', 'player_5_objectId', 'player_5_jerseyNum', 'player_5_x', 'player_5_y',
                 'player_6_teamId', 'player_6_objectId', 'player_6_jerseyNum', 'player_6_x', 'player_6_y',
                 'player_7_teamId', 'player_7_objectId', 'player_7_jerseyNum', 'player_7_x', 'player_7_y',
                 'player_8_teamId', 'player_8_objectId', 'player_8_jerseyNum', 'player_8_x', 'player_8_y',
                 'player_9_teamId', 'player_9_objectId', 'player_9_jerseyNum', 'player_9_x', 'player_9_y',
                 'player_10_teamId', 'player_10_objectId', 'player_10_jerseyNum', 'player_10_x', 'player_10_y',
                 'player_11_teamId', 'player_11_objectId', 'player_11_jerseyNum', 'player_11_x', 'player_11_y',
                 'player_12_teamId', 'player_12_objectId', 'player_12_jerseyNum', 'player_12_x', 'player_12_y',
                 'player_13_teamId', 'player_13_objectId', 'player_13_jerseyNum', 'player_13_x', 'player_13_y',
                 'player_14_teamId', 'player_14_objectId', 'player_14_jerseyNum', 'player_14_x', 'player_14_y',
                 'player_15_teamId', 'player_15_objectId', 'player_15_jerseyNum', 'player_15_x', 'player_15_y',
                 'player_16_teamId', 'player_16_objectId', 'player_16_jerseyNum', 'player_16_x', 'player_16_y',
                 'player_17_teamId', 'player_17_objectId', 'player_17_jerseyNum', 'player_17_x', 'player_17_y',
                 'player_18_teamId', 'player_18_objectId', 'player_18_jerseyNum', 'player_18_x', 'player_18_y',
                 'player_19_teamId', 'player_19_objectId', 'player_19_jerseyNum', 'player_19_x', 'player_19_y',
                 'player_20_teamId', 'player_20_objectId', 'player_20_jerseyNum', 'player_20_x', 'player_20_y',
                 'player_21_teamId', 'player_21_objectId', 'player_21_jerseyNum', 'player_21_x', 'player_21_y',
                 'player_22_teamId', 'player_22_objectId', 'player_22_jerseyNum', 'player_22_x', 'player_22_y',
                 'to_del_1',
                 'ball_x', 'ball_y', 'ball_z',
                 'to_del_2']
filename = f'tracking-data-{match_id}-25fps.txt'
tracking = pd.read_csv(path + filename, sep=";|,|:", names=tracking_cols, header=None, engine='python')
tracking = tracking.drop(labels=['to_del_1', 'to_del_2'], axis=1)

temp = tracking[tracking['half_indicator'] == 1]['time_of_current_half'].max()
tracking['timeMilliSec'] = tracking['time_of_current_half'] + temp * (tracking['half_indicator'] - 1)


with open(path + filename_event) as f:
    d = json.load(f)


def jsonNormalize(data):
    dic_flattened = (flatten(dd) for dd in data)
    df = pd.DataFrame(dic_flattened)
    return df


df1 = jsonNormalize(d['liveData']['event'])
interesting_periods = [1, 2, 3, 4, 5]
interesting_event_ids = [1, 2, 3, 7, 8, 10, 11, 13, 14, 15, 16, 74]
df2 = df1[df1['typeId'].apply(lambda x: x == 32)]
df1 = df1[df1['periodId'].apply(lambda x: x in interesting_periods)]
df1 = df1[df1['typeId'].apply(lambda x: x in interesting_event_ids)]

id_mapping = pd.read_csv(path + filename_metadata)
id_mapping = id_mapping[['matchName', 'stats_id']]
id_mapping.columns = ['playerName', 'playerTrackingId']


def get_tr_id(playerNames):
    ids = []
    for playerName in playerNames:
        id = int(id_mapping[id_mapping['playerName'] == playerName]['playerTrackingId'].values[0])
        ids.append(id)
    return ids


df1['playerTrackingId'] = get_tr_id(df1['playerName'].values)

# Calculate remaining time from half
# Need to group by period because extra time may differ

df1['to_sec'] = df1['timeMin'] * 60 + df1['timeSec']
df1['sec_remaining'] = df1.groupby('periodId').to_sec.transform('max') - df1.to_sec

temp = df1.iloc[0]['timeStamp'].replace('Z', '-01')
temp2 = df1[(df1['periodId'] == 2) & (df1['to_sec'] == 45 * 60)]['timeStamp'].values[0].replace('Z', '-01')
OFFSET_1 = tracking.iloc[0]['timestamp'] - parser.parse(temp).timestamp() * 1000
OFFSET_2 = tracking[(tracking['half_indicator'] == 2) & (tracking['time_of_current_half'] == 0)]['timestamp'].values[
               0] - parser.parse(temp2).timestamp() * 1000


# syncs the tracking and event timestamps for better matching later
def getmillisecs(timestamps, half):
    out = []
    for timestamp in timestamps:
        timestamp = timestamp.replace('Z', '-01')
        if half == 1:
            yourdate = parser.parse(timestamp).timestamp() * 1000 + OFFSET_1
        else:
            yourdate = parser.parse(timestamp).timestamp() * 1000 + OFFSET_2
        out.append(yourdate)
    return out


firsthalf = df1.loc[df1['periodId'] == 1]['timeStamp'].values.tolist()
secondhalf = df1.loc[df1['periodId'] == 2]['timeStamp'].values.tolist()
first = getmillisecs(firsthalf, 1)
second = getmillisecs(secondhalf, 2)
timestamps = first + second
df1['timeStamp'] = timestamps

matchinfo = jsonNormalize(d['matchInfo']['contestant'])

home_team_id = ''

if matchinfo.loc[0,'position'] == 'home':
    df1.loc[df1.contestantId == matchinfo.loc[0,'id'],'event_team'] = 0
    home_team_id = matchinfo.loc[0,'id']
    df1.loc[df1.contestantId == matchinfo.loc[1,'id'],'event_team'] = 1
else:
    df1.loc[df1.contestantId == matchinfo.loc[0,'id'],'event_team'] = 1
    df1.loc[df1.contestantId == matchinfo.loc[1,'id'],'event_team'] = 0
    home_team_id = matchinfo.loc[1,'id']
    
df1['home_attack_dir'] = df2.loc[df2.contestantId == home_team_id,'qualifier_0_value'].values[0]


half1_team1_attackdir = df2['qualifier_0_value'].iloc[0]  # 'Right to Left' or 'Left to Right'

df1['Pass_end_x'] = np.nan
df1['Pass_end_y'] = np.nan
q_ids=[]
q_vals=[]
for column in df1:
    if column.startswith('qualifier') and column.endswith('qualifierId'):
        q_ids.append(column.split('_')[1])
    elif column.startswith('qualifier') and column.endswith('value'):
        q_vals.append(column.split('_')[1])
qs=list(set(q_ids).intersection(q_vals))

if (half1_team1_attackdir == 'Right to Left'):
    df1.loc[(df1['periodId'] == 1) & (df1['contestantId'] == df2['contestantId'].iloc[0]), ['x']] = 100 - df1['x']
    df1.loc[(df1['periodId'] == 2) & (df1['contestantId'] == df2['contestantId'].iloc[1]), ['x']] = 100 - df1['x']
    df1.loc[(df1['periodId'] == 1) & (df1['contestantId'] == df2['contestantId'].iloc[0]), ['y']] = 100 - df1['y']
    df1.loc[(df1['periodId'] == 2) & (df1['contestantId'] == df2['contestantId'].iloc[1]), ['y']] = 100 - df1['y']
    # transform the pass destination too
    # count qualifiers

    # X:
    for colnum in qs:
        df1.loc[(df1[f'qualifier_{colnum}_qualifierId'] == 140) &
                (df1['periodId'] == 1) & (df1['contestantId'] == df2['contestantId'].iloc[0]),
                ['Pass_end_x']] = (df1[f'qualifier_{colnum}_value'])

        df1.loc[(df1[f'qualifier_{colnum}_qualifierId'] == 140) &
                (df1['periodId'] == 2) & (df1['contestantId'] == df2['contestantId'].iloc[1]),
                ['Pass_end_x']] = (df1[f'qualifier_{colnum}_value'])
        # Y:
        df1.loc[(df1[f'qualifier_{colnum}_qualifierId'] == 141) &
                (df1['periodId'] == 1) & (df1['contestantId'] == df2['contestantId'].iloc[0]),
                ['Pass_end_y']] = (df1[f'qualifier_{colnum}_value'])

        df1.loc[(df1[f'qualifier_{colnum}_qualifierId'] == 141) &
                (df1['periodId'] == 2) & (df1['contestantId'] == df2['contestantId'].iloc[1]),
                ['Pass_end_y']] = (df1[f'qualifier_{colnum}_value'])

elif (half1_team1_attackdir == 'Left to Right'):
    df1.loc[(df1['periodId'] == 1) & (df1['contestantId'] == df2['contestantId'].iloc[1]), ['x']] = 100 - df1['x']
    df1.loc[(df1['periodId'] == 2) & (df1['contestantId'] == df2['contestantId'].iloc[0]), ['x']] = 100 - df1['x']
    df1.loc[(df1['periodId'] == 1) & (df1['contestantId'] == df2['contestantId'].iloc[1]), ['y']] = 100 - df1['y']
    df1.loc[(df1['periodId'] == 2) & (df1['contestantId'] == df2['contestantId'].iloc[0]), ['y']] = 100 - df1['y']

    # transform the pass destination too
    # X:
    for colnum in qs:
        df1.loc[(df1[f'qualifier_{colnum}_qualifierId'] == 140) &  # here
                (df1['periodId'] == 1) & (df1['contestantId'] == df2['contestantId'].iloc[1]),
                ['Pass_end_x']] = (df1[f'qualifier_{colnum}_value'])

        df1.loc[(df1[f'qualifier_{colnum}_qualifierId'] == 140) &
                (df1['periodId'] == 2) & (df1['contestantId'] == df2['contestantId'].iloc[0]),
                ['Pass_end_x']] = (df1[f'qualifier_{colnum}_value'])
        # Y:
        df1.loc[(df1[f'qualifier_{colnum}_qualifierId'] == 141) &
                (df1['periodId'] == 1) & (df1['contestantId'] == df2['contestantId'].iloc[1]),
                ['Pass_end_y']] = (df1[f'qualifier_{colnum}_value'])

        df1.loc[(df1[f'qualifier_{colnum}_qualifierId'] == 141) &
                (df1['periodId'] == 2) & (df1['contestantId'] == df2['contestantId'].iloc[0]),
                ['Pass_end_y']] = (df1[f'qualifier_{colnum}_value'])

# Pass_end_x and Pass_end_y now only contains the values that need to be flipped, so:
df1['Pass_end_x'] = pd.to_numeric(df1.Pass_end_x, errors='coerce')
df1['Pass_end_y'] = pd.to_numeric(df1.Pass_end_y, errors='coerce')

df1['Pass_end_x'] = 100 - df1['Pass_end_x']
df1['Pass_end_y'] = 100 - df1['Pass_end_y']

# Now add the ones that will not need flipping.
if (half1_team1_attackdir == 'Right to Left'):

    # X: (do we need typeId too??)
    for colnum in qs:
        df1.loc[(df1[f'qualifier_{colnum}_qualifierId'] == 140) &
                (df1['periodId'] == 1) & (df1['contestantId'] == df2['contestantId'].iloc[1]),
                ['Pass_end_x']] = (df1[f'qualifier_{colnum}_value'])

        df1.loc[(df1[f'qualifier_{colnum}_qualifierId'] == 140) &
                (df1['periodId'] == 2) & (df1['contestantId'] == df2['contestantId'].iloc[0]),
                ['Pass_end_x']] = (df1[f'qualifier_{colnum}_value'])
        # Y:
        df1.loc[(df1[f'qualifier_{colnum}_qualifierId'] == 141) &
                (df1['periodId'] == 1) & (df1['contestantId'] == df2['contestantId'].iloc[1]),
                ['Pass_end_y']] = (df1[f'qualifier_{colnum}_value'])

        df1.loc[(df1[f'qualifier_{colnum}_qualifierId'] == 141) &
                (df1['periodId'] == 2) & (df1['contestantId'] == df2['contestantId'].iloc[0]),
                ['Pass_end_y']] = (df1[f'qualifier_{colnum}_value'])

elif (half1_team1_attackdir == 'Left to Right'):
    # X:
    for colnum in qs:
        df1.loc[(df1[f'qualifier_{colnum}_qualifierId'] == 140) &
                (df1['periodId'] == 1) & (df1['contestantId'] == df2['contestantId'].iloc[0]),
                ['Pass_end_x']] = (df1[f'qualifier_{colnum}_value'])

        df1.loc[(df1[f'qualifier_{colnum}_qualifierId'] == 140) &
                (df1['periodId'] == 2) & (df1['contestantId'] == df2['contestantId'].iloc[1]),
                ['Pass_end_x']] = (df1[f'qualifier_{colnum}_value'])
        # Y:
        df1.loc[(df1[f'qualifier_{colnum}_qualifierId'] == 141) &
                (df1['periodId'] == 1) & (df1['contestantId'] == df2['contestantId'].iloc[0]),
                ['Pass_end_y']] = (df1[f'qualifier_{colnum}_value'])

        df1.loc[(df1[f'qualifier_{colnum}_qualifierId'] == 141) &
                (df1['periodId'] == 2) & (df1['contestantId'] == df2['contestantId'].iloc[1]),
                ['Pass_end_y']] = (df1[f'qualifier_{colnum}_value'])

df1['Pass_end_x'] = pd.to_numeric(df1.Pass_end_x, errors='coerce')
df1['Pass_end_y'] = pd.to_numeric(df1.Pass_end_y, errors='coerce')
# Now scale from 100x100 to 105x68


df1['x'] = df1['x'] * 1.05
df1['y'] = df1['y'] * 0.68
df1['Pass_end_x'] = df1['Pass_end_x'] * 1.05
df1['Pass_end_y'] = df1['Pass_end_y'] * 0.68

# tracking flip y
tracking['ball_y'] = 68 - tracking['ball_y']
for num in range(1, 23):
    tracking[f'player_{num}_y'] = 68 - tracking[f'player_{num}_y']

# calculating velocity (speed and direction) - for players
dT = tracking.loc[1,'timestamp']-tracking.loc[0,'timestamp']
MAXSPEED = 12
MOVING_WINDOW =7
#ma_window = np.ones( MOVING_WINDOW ) / MOVING_WINDOW 

for p_num in range(1,23):
    # directions
    Vx = tracking[f'player_{p_num}_x'].diff() / dT
    Vy = tracking[f'player_{p_num}_y'].diff() / dT

    # get rid of outliers
    if MAXSPEED > 0:
        raw_speed = np.sqrt((Vx**2)+(Vy**2))
        Vx[ raw_speed>MAXSPEED ] = np.nan
        Vy[ raw_speed>MAXSPEED ] = np.nan

    # smoothing
    # calculate first half velocity
    #Vx = np.convolve( Vx , ma_window, mode='same' ) 
    #Vy = np.convolve( Vy , ma_window, mode='same' )      
    Vx = Vx.rolling(MOVING_WINDOW,min_periods = 0,center=False).mean()
    Vy = Vy.rolling(MOVING_WINDOW,min_periods = 0,center=False).mean()

    # apply speed and direction values
    tracking[f'player_{p_num}_direction_x'] = Vx
    tracking[f'player_{p_num}_direction_y'] = Vy
    tracking[f'player_{p_num}_speed'] = np.sqrt((Vx**2)+(Vy**2))
    tracking.loc[:,[f'player_{p_num}_direction_x',f'player_{p_num}_direction_y',f'player_{p_num}_speed']] =\
         tracking.loc[:,[f'player_{p_num}_direction_x',f'player_{p_num}_direction_y',f'player_{p_num}_speed']].fillna(0)

# calculating velocity (speed and direction) - for ball
Vbx = tracking['ball_x'].diff() / dT
Vby = tracking['ball_y'].diff() / dT
if MAXSPEED > 0:
    raw_speed = np.sqrt(Vbx**2+Vby**2)
    Vbx[ raw_speed>MAXSPEED ] = np.nan
    Vby[ raw_speed>MAXSPEED ] = np.nan
Vbx = Vbx.rolling(MOVING_WINDOW,min_periods = 0, center= False).mean()
Vby = Vby.rolling(MOVING_WINDOW,min_periods = 0, center= False).mean()
tracking['ball_direction_x'] = Vx
tracking['ball_direction_y'] = Vy
tracking['ball_speed'] = np.sqrt((Vx**2)+(Vy**2))

tracking.loc[:,['ball_direction_x','ball_direction_y','ball_speed']] = tracking.loc[:,['ball_direction_x','ball_direction_y','ball_speed']].fillna(0)
# Features for events

# For goaldifference; (matchinfo is created to get contestant_id)
homeId = matchinfo['id'].iloc[0]
awayId = matchinfo['id'].iloc[1]

goaldiff_df = pd.DataFrame(None)
goaldiff_df['typeId'] = df1['typeId']
goaldiff_df['contestantId'] = df1['contestantId']
goaldiff_df['is_home_goal'] = np.where((goaldiff_df['typeId'] != 16) | (goaldiff_df['contestantId'] != homeId), 0, 1)
goaldiff_df['is_away_goal'] = np.where((goaldiff_df['typeId'] != 16) | (goaldiff_df['contestantId'] != awayId), 0, 1)
goaldiff_df['Curr_home_goals'] = goaldiff_df['is_home_goal'].cumsum()
goaldiff_df['Curr_away_goals'] = goaldiff_df['is_away_goal'].cumsum()
goaldiff_df['Goal_difference_home'] = goaldiff_df['Curr_home_goals'] - goaldiff_df['Curr_away_goals']
goaldiff_df['Goal_difference_away'] = goaldiff_df['Goal_difference_home'] * -1

df1['Goal_difference_home'] = goaldiff_df['Goal_difference_home']
df1['Goal_difference_away'] = df1['Goal_difference_home'] * -1

tracking.to_csv(path + f'flat-tracking-{match_id}-25fps.csv', index=False)

df1.to_csv(path + f'events-ma13-with-features-{match_id}.csv', index=False)