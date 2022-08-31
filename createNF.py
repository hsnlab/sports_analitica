import json
import pandas as pd
import numpy as np
from flatten_json import flatten
from dateutil import parser
from math import sqrt
from statistics import mean
from timeit import default_timer as timer
import os
import csv


def jsonNormalize(data):
    dic_flattened = (flatten(dd) for dd in data)
    df = pd.DataFrame(dic_flattened)
    return df


def get_tr_id(id_mapping, playerNames):
    ids = []
    for playerName in playerNames:
        id = id_mapping[id_mapping['playerName'] == playerName]['playerTrackingId'].values[0]
        ids.append(id)
    return ids


# syncs the tracking and event timestamps for better matching later
def getmillisecs(timestamps, offset):
    out = []
    for timestamp in timestamps:
        timestamp = timestamp.replace('Z', '-01')
        yourdate = int(parser.parse(timestamp).timestamp() * 1000 + offset)
        out.append(yourdate)
    return out


def flatten_and_features(path, match_id, metadata_fn):
    filename_event = f'events-ma13-{match_id}.json'
    filename_metadata = metadata_fn

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
    try:
        tracking = pd.read_csv(path + filename, sep=";|,|:", names=tracking_cols, header=None, engine='python')
        tracking = tracking.drop(labels=['to_del_1', 'to_del_2'], axis=1)

        temp = tracking[tracking['half_indicator'] == 1]['time_of_current_half'].max()
        tracking['timeMilliSec'] = tracking['time_of_current_half'] + temp * (tracking['half_indicator'] - 1)
    except:
        return pd.DataFrame(), pd.DataFrame()

    with open(path + filename_event) as f:
        d = json.load(f)


    df1 = jsonNormalize(d['liveData']['event'])
    interesting_periods = [1, 2, 3, 4, 5]
    interesting_event_ids = [1, 2, 3, 5, 7, 8, 12, 13, 14, 15, 16, 74]
    df2 = df1[df1['typeId'].apply(lambda x: x == 32)]
    df1 = df1[df1['periodId'].apply(lambda x: x in interesting_periods)]
    df1 = df1[df1['typeId'].apply(lambda x: x in interesting_event_ids)]
    df1 = df1[pd.notna(df1['playerName'])]

    id_mapping = pd.read_csv(path + filename_metadata)
    id_mapping = id_mapping[['matchName', 'stats_id']]
    id_mapping.columns = ['playerName', 'playerTrackingId']

    df1['playerTrackingId'] = get_tr_id(id_mapping, df1['playerName'].values)

    # Calculate remaining time from half
    # Need to group by period because extra time may differ

    df1['to_sec'] = df1['timeMin'] * 60 + df1['timeSec']
    df1['sec_remaining'] = df1.groupby('periodId').to_sec.transform('max') - df1.to_sec

    temp = df1.iloc[0]['timeStamp'].replace('Z', '-01')
    temp2 = df1[(df1['periodId'] == 2)]['timeStamp'].values[0].replace('Z', '-01')
    OFFSET_1 = tracking.iloc[0]['timestamp'] - parser.parse(temp).timestamp() * 1000
    OFFSET_2 = \
    tracking[(tracking['half_indicator'] == 2) & (tracking['time_of_current_half'] == 0)]['timestamp'].values[
        0] - parser.parse(temp2).timestamp() * 1000

    firsthalf = df1.loc[df1['periodId'] == 1]['timeStamp'].values.tolist()
    secondhalf = df1.loc[df1['periodId'] == 2]['timeStamp'].values.tolist()
    first = getmillisecs(firsthalf, OFFSET_1)
    second = getmillisecs(secondhalf, OFFSET_2)
    timestamps = first + second
    df1['timeStamp'] = timestamps

    matchinfo = jsonNormalize(d['matchInfo']['contestant'])

    home_team_id = ''
    if matchinfo.loc[0, 'position'] == 'home':
        df1.loc[df1.contestantId == matchinfo.loc[0, 'id'], 'event_team'] = 0
        home_team_id = matchinfo.loc[0, 'id']
        df1.loc[df1.contestantId == matchinfo.loc[1, 'id'], 'event_team'] = 1
    else:
        df1.loc[df1.contestantId == matchinfo.loc[0, 'id'], 'event_team'] = 1
        df1.loc[df1.contestantId == matchinfo.loc[1, 'id'], 'event_team'] = 0
        home_team_id = matchinfo.loc[1, 'id']

    df1['home_attack_dir'] = df2.loc[df2.contestantId == home_team_id, 'qualifier_0_value'].values[0]

    half1_team1_attackdir = df2['qualifier_0_value'].iloc[0]  # 'Right to Left' or 'Left to Right'
    df1['Pass_end_x'] = np.nan
    df1['Pass_end_y'] = np.nan
    q_ids = []
    q_vals = []
    for column in df1:
        if column.startswith('qualifier') and column.endswith('qualifierId'):
            q_ids.append(column.split('_')[1])
        elif column.startswith('qualifier') and column.endswith('value'):
            q_vals.append(column.split('_')[1])
    qs = list(set(q_ids).intersection(q_vals))

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
    df1['Pass_end_x'] = df1['Pass_end_x'].fillna(0)
    df1['Pass_end_y'] = pd.to_numeric(df1.Pass_end_y, errors='coerce')
    df1['Pass_end_y'] = df1['Pass_end_y'].fillna(0)
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
    dT = tracking.loc[1, 'timestamp'] - tracking.loc[0, 'timestamp']
    MAXSPEED = 12
    MOVING_WINDOW = 7
    # ma_window = np.ones( MOVING_WINDOW ) / MOVING_WINDOW

    for p_num in range(1, 23):
        # directions
        Vx = tracking[f'player_{p_num}_x'].diff() / dT
        Vy = tracking[f'player_{p_num}_y'].diff() / dT

        # get rid of outliers
        if MAXSPEED > 0:
            raw_speed = np.sqrt((Vx ** 2) + (Vy ** 2))
            Vx[raw_speed > MAXSPEED] = np.nan
            Vy[raw_speed > MAXSPEED] = np.nan

        # smoothing
        # calculate first half velocity
        # Vx = np.convolve( Vx , ma_window, mode='same' )
        # Vy = np.convolve( Vy , ma_window, mode='same' )
        Vx = Vx.rolling(MOVING_WINDOW, min_periods=0, center=False).mean()
        Vy = Vy.rolling(MOVING_WINDOW, min_periods=0, center=False).mean()

        # apply speed and direction values
        tracking[f'player_{p_num}_direction_x'] = Vx
        tracking[f'player_{p_num}_direction_y'] = Vy
        tracking[f'player_{p_num}_speed'] = np.sqrt((Vx ** 2) + (Vy ** 2))
        tracking.loc[:, [f'player_{p_num}_direction_x', f'player_{p_num}_direction_y', f'player_{p_num}_speed']] = \
            tracking.loc[:,
            [f'player_{p_num}_direction_x', f'player_{p_num}_direction_y', f'player_{p_num}_speed']].fillna(0)

    # calculating velocity (speed and direction) - for ball
    Vbx = tracking['ball_x'].diff() / dT
    Vby = tracking['ball_y'].diff() / dT
    if MAXSPEED > 0:
        raw_speed = np.sqrt(Vbx ** 2 + Vby ** 2)
        Vbx[raw_speed > MAXSPEED] = np.nan
        Vby[raw_speed > MAXSPEED] = np.nan
    Vbx = Vbx.rolling(MOVING_WINDOW, min_periods=0, center=False).mean()
    Vby = Vby.rolling(MOVING_WINDOW, min_periods=0, center=False).mean()
    tracking['ball_direction_x'] = Vx
    tracking['ball_direction_y'] = Vy
    tracking['ball_speed'] = np.sqrt((Vx ** 2) + (Vy ** 2))

    tracking.loc[:, ['ball_direction_x', 'ball_direction_y', 'ball_speed']] = tracking.loc[:,
                                                                              ['ball_direction_x', 'ball_direction_y',
                                                                               'ball_speed']].fillna(0)

    # Features for events
    # For goaldifference; (matchinfo is created to get contestant_id)
    homeId = matchinfo['id'].iloc[0]
    awayId = matchinfo['id'].iloc[1]

    goaldiff_df = pd.DataFrame(None)
    goaldiff_df['typeId'] = df1['typeId']
    goaldiff_df['contestantId'] = df1['contestantId']
    goaldiff_df['is_home_goal'] = np.where((goaldiff_df['typeId'] != 16) | (goaldiff_df['contestantId'] != homeId), 0,
                                           1)
    goaldiff_df['is_away_goal'] = np.where((goaldiff_df['typeId'] != 16) | (goaldiff_df['contestantId'] != awayId), 0,
                                           1)
    goaldiff_df['Curr_home_goals'] = goaldiff_df['is_home_goal'].cumsum()
    goaldiff_df['Curr_away_goals'] = goaldiff_df['is_away_goal'].cumsum()
    goaldiff_df['Goal_difference_home'] = goaldiff_df['Curr_home_goals'] - goaldiff_df['Curr_away_goals']
    goaldiff_df['Goal_difference_away'] = goaldiff_df['Goal_difference_home'] * -1

    df1['Goal_difference_home'] = goaldiff_df['Goal_difference_home']
    df1['Goal_difference_away'] = df1['Goal_difference_home'] * -1

    return tracking, df1


# helper functions for calculating disance between timestamps
def distance_time(time_from, time_to):
    return abs(time_to - time_from)


def conv_milisec_to_sec(milliseconds):
    return round((milliseconds) / 1000)


def conv_min_sec_to_sec(minute, second):
    return (minute * 60 + second)


# helper functions for calculating distance between positions
def distance_pos(x_1, y_1, x_2, y_2):
    return sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)


def get_player_pos(playerId, frame):
    x = 0
    y = 0
    values = frame.values.tolist()
    for elem in values:
        i = values.index(elem)
        if elem == playerId:
            x = frame[i + 2]
            y = frame[i + 3]
    return x, y


# calculate distance metric for an event and trackinf frame pair
def calc_distance(event, frame, justTime):
    if not justTime:
        # calc time distance
        ev_time = int(40 * round(float(event['timeStamp']) / 40))
        fr_time = frame['timestamp']
        time_dist = distance_time(ev_time, fr_time)
        # calc positional distance
        #   -ball
        pos_distance_ball = distance_pos(event['x'], event['y'], frame['ball_x'], frame['ball_y'])
        #   -player performing the event
        player = event['playerTrackingId']
        player_x, player_y = get_player_pos(player, frame)
        pos_distance_player = distance_pos(event['x'], event['y'], player_x, player_y)
        # return overall distance
        dist = time_dist * 0.7 + pos_distance_ball * 0.15 + pos_distance_player * 0.15
    else:
        # calc time distance
        ev_time = int(40 * round(float(event['timeStamp']) / 40))
        fr_time = frame['timestamp']
        dist = distance_time(ev_time, fr_time)
    return (dist)


# evaluates the syncing - with the difference between tracking player position and event position
def eval(event_df, tracking_df):
    distances = []
    distances_ball = []
    distances_x = []
    distances_y = []
    for ev_i, event in event_df.iterrows():
        frame = tracking_df.iloc[event['frame_id']]
        player_x, player_y = get_player_pos(event['playerTrackingId'], frame)
        distance = distance_pos(event['x'], event['y'], player_x, player_y)
        distances_ball.append(distance_pos(event['x'], event['y'], frame['ball_x'], frame['ball_y']))
        distances_x.append(abs(event['x'] - player_x))
        distances_y.append(abs(event['y'] - player_y))
        distances.append(distance)
    return distances, distances_ball, distances_x, distances_y


def match_events_25fps(event_df, tracking_df, justTime=True):
    frame_ids = []
    for ev_i, event in event_df.iterrows():
        rounded_time = int(40 * round(float(event['timeStamp']) / 40))
        fr_id_l = tracking_df.index[tracking_df['timestamp'] == rounded_time].tolist()
        if fr_id_l:
            frame_ids.append(fr_id_l[0])
        else:
            print(f"error, real time: {event['timeStamp']}, rounded time: {rounded_time}")
    return frame_ids


def match_events_10fps(event_df, tracking_df, justTime=True):
    frame_ids = []
    for ev_i, event in event_df.iterrows():
        rounded_time = int(100 * round(float(event['timeStamp']) / 100))
        fr_id_l = tracking_df.index[tracking_df['timestamp'] == rounded_time].tolist()
        if fr_id_l:
            frame_ids.append(fr_id_l[0])
        else:
            print(f"error, real time: {event['timeStamp']}, rounded time: {rounded_time}")
    return frame_ids


GOAL_X = 100
GOAL_Y = 50
GOALPOST_1Y = 55.3  # was 53.66
GOALPOST_2Y = 44.7  # was 46.34
DIST_GOALPOSTS = 10.6  # a;  was 7.22


def calc_angle_and_distance_to_goal(X, Y, needs_flipping):
    if needs_flipping:
        X = 105 - X
    X = X / 1.05
    Y = Y / 0.68
    diff_X = abs(GOAL_X - X)
    diff_Y = abs(GOAL_Y - Y)
    dist_to_goal = np.sqrt(diff_X ** 2 + diff_Y ** 2)

    diff_gp1y = abs(GOALPOST_1Y - Y)
    diff_gp2y = abs(GOALPOST_2Y - Y)
    dist_gp1y = np.sqrt(diff_X ** 2 + diff_gp1y ** 2)  # b
    dist_gp2y = np.sqrt(diff_X ** 2 + diff_gp2y ** 2)  # c
    ang_to_goal = np.arccos((dist_gp1y ** 2 + dist_gp2y ** 2 - DIST_GOALPOSTS ** 2) / (2 * dist_gp1y * dist_gp2y))

    return dist_to_goal, ang_to_goal


def sync_and_features(tracking_df, event_df, is25fps=True):
    home_attack_dir = event_df['home_attack_dir'].values[0]
    if is25fps:
        frame_ids = match_events_25fps(event_df, tracking_df)
    else:
        frame_ids = match_events_10fps(event_df, tracking_df)
    try:
        event_df['frame_id'] = frame_ids
    except ValueError:
        print('Value error while syncing')
        return pd.DataFrame()
    # event_df['frame_id'] = np.nan
    # event_df.iloc[:len(frame_ids)]['frame_id']=frame_ids
    # event_df = event_df[pd.notna(event_df['frame_id'])]
    dist, dist_b, dist_x, dist_y = eval(event_df, tracking_df)
    event_df['distance'] = dist
    event_df = event_df[event_df['distance'] < 15]

    tracking_df['frame_id'] = tracking_df.index

    tracking_w_events = pd.merge(tracking_df, event_df, how='outer', on=['frame_id'], suffixes=('_tracking', '_event'))
    synced_events = tracking_w_events[pd.notna(tracking_w_events['eventId'])].copy()
    for ev_idx, event in synced_events.iterrows():
        ball_carrier = event['playerTrackingId']
        bc_x, bc_y = get_player_pos(ball_carrier, event)

        for num in range(1, 23):
            p_x, p_y = event[f'player_{num}_x'], event[f'player_{num}_y']
            dist = distance_pos(bc_x, bc_y, p_x, p_y)
            if dist != 0:
                synced_events.loc[ev_idx, f'player_{num}_dist_to_bc'] = dist
                synced_events.loc[ev_idx, f'player_{num}_is_bc'] = 0
            else:
                synced_events.loc[ev_idx, f'player_{num}_dist_to_bc'] = dist
                synced_events.loc[ev_idx, f'player_{num}_is_bc'] = 1
                synced_events.loc[ev_idx, 'event_team'] = event[f'player_{num}_teamId']

        # ball
        ball_x, ball_y = event['ball_x'], event['ball_y']
        dist_b = distance_pos(bc_x, bc_y, ball_x, ball_y)
        synced_events.loc[ev_idx, 'ball_dist_to_bc'] = dist_b
        synced_events.loc[ev_idx, 'ball_is_bc'] = 0

    # ball and player distance- and angle to goal
    second_half_idx = synced_events.half_indicator.idxmax()
    if home_attack_dir == 'Left to Right':
        # away_players - calc_angle_and_distance_to_goal(away player X, away player Y, False)
        for num in range(11, 21):
            a_p_X_fh = synced_events.loc[:second_half_idx, f'player_{num}_x']
            a_p_Y_fh = synced_events.loc[:second_half_idx, f'player_{num}_y']
            a_p_X_sh = synced_events.loc[second_half_idx:, f'player_{num}_x']
            a_p_Y_sh = synced_events.loc[second_half_idx:, f'player_{num}_y']

            synced_events.loc[:second_half_idx, f'player_{num}_dist_to_goal'], \
            synced_events.loc[:second_half_idx, f'player_{num}_angle_to_goal'] = calc_angle_and_distance_to_goal(
                a_p_X_fh, a_p_Y_fh, True)
            synced_events.loc[second_half_idx:, f'player_{num}_dist_to_goal'], \
            synced_events.loc[second_half_idx:, f'player_{num}_angle_to_goal'] = calc_angle_and_distance_to_goal(
                a_p_X_sh, a_p_Y_sh, False)
        # away goalkeeper
        a_gk_X_fh = synced_events.loc[:second_half_idx, 'player_22_x']
        a_gk_Y_fh = synced_events.loc[:second_half_idx, 'player_22_y']
        a_gk_X_sh = synced_events.loc[second_half_idx:, 'player_22_x']
        a_gk_Y_sh = synced_events.loc[second_half_idx:, 'player_22_y']
        synced_events.loc[:second_half_idx, 'player_22_dist_to_goal'], \
        synced_events.loc[:second_half_idx, 'player_22_angle_to_goal'] = calc_angle_and_distance_to_goal(a_gk_X_fh,
                                                                                                         a_gk_Y_fh,
                                                                                                         True)
        synced_events.loc[second_half_idx:, 'player_22_dist_to_goal'], \
        synced_events.loc[second_half_idx:, 'player_22_angle_to_goal'] = calc_angle_and_distance_to_goal(a_gk_X_sh,
                                                                                                         a_gk_Y_sh,
                                                                                                         False)
        # home players
        for num in range(1, 11):
            h_p_X_fh = synced_events.loc[:second_half_idx, f'player_{num}_x']
            h_p_Y_fh = synced_events.loc[:second_half_idx, f'player_{num}_y']
            h_p_X_sh = synced_events.loc[second_half_idx:, f'player_{num}_x']
            h_p_Y_sh = synced_events.loc[second_half_idx:, f'player_{num}_y']

            synced_events.loc[:second_half_idx, f'player_{num}_dist_to_goal'], \
            synced_events.loc[:second_half_idx, f'player_{num}_angle_to_goal'] = calc_angle_and_distance_to_goal(
                h_p_X_fh, h_p_Y_fh, False)
            synced_events.loc[second_half_idx:, f'player_{num}_dist_to_goal'], \
            synced_events.loc[second_half_idx:, f'player_{num}_angle_to_goal'] = calc_angle_and_distance_to_goal(
                h_p_X_sh, h_p_Y_sh, True)
        # hom
        # home goalkeeper - calc_angle_and_distance_to_goal(home player X, home player Y, True)
        h_gk_X_fh = synced_events.loc[:second_half_idx, 'player_21_x']
        h_gk_Y_fh = synced_events.loc[:second_half_idx, 'player_21_y']
        h_gk_X_sh = synced_events.loc[second_half_idx:, 'player_21_x']
        h_gk_Y_sh = synced_events.loc[second_half_idx:, 'player_21_y']
        synced_events.loc[:second_half_idx, 'player_21_dist_to_goal'], \
        synced_events.loc[:second_half_idx, 'player_21_angle_to_goal'] = calc_angle_and_distance_to_goal(h_gk_X_fh,
                                                                                                         h_gk_Y_fh,
                                                                                                         False)
        synced_events.loc[second_half_idx:, 'player_21_dist_to_goal'], \
        synced_events.loc[second_half_idx:, 'player_21_angle_to_goal'] = calc_angle_and_distance_to_goal(h_gk_X_sh,
                                                                                                         h_gk_Y_sh,
                                                                                                         True)

        # ball
        b_X_fh = synced_events.loc[:second_half_idx, 'ball_x']
        b_Y_fh = synced_events.loc[:second_half_idx, 'ball_y']
        b_X_sh = synced_events.loc[second_half_idx:, 'ball_x']
        b_Y_sh = synced_events.loc[second_half_idx:, 'ball_y']

        if event['event_team'] == 0:
            synced_events.loc[:second_half_idx, 'ball_dist_to_goal'], \
            synced_events.loc[:second_half_idx, 'ball_angle_to_goal'] = calc_angle_and_distance_to_goal(b_X_fh, b_Y_fh,
                                                                                                        False)
            synced_events.loc[second_half_idx:, 'ball_dist_to_goal'], \
            synced_events.loc[second_half_idx:, 'ball_angle_to_goal'] = calc_angle_and_distance_to_goal(b_X_sh, b_Y_sh,
                                                                                                        True)
            # calc_angle_and_distance_to_goal(ball X, ball Y, True)
        else:
            synced_events.loc[:second_half_idx, 'ball_dist_to_goal'], \
            synced_events.loc[:second_half_idx, 'ball_angle_to_goal'] = calc_angle_and_distance_to_goal(b_X_fh, b_Y_fh,
                                                                                                        True)
            synced_events.loc[second_half_idx:, 'ball_dist_to_goal'], \
            synced_events.loc[second_half_idx:, 'ball_angle_to_goal'] = calc_angle_and_distance_to_goal(b_X_sh, b_Y_sh,
                                                                                                        False)
            # calc_angle_and_distance_to_goal(ball X, ball Y, False)

    elif home_attack_dir == 'Right to Left':
        # away_players - calc_angle_and_distance_to_goal(away player X, away player Y, False)
        for num in range(11, 21):
            a_p_X_fh = synced_events.loc[:second_half_idx, f'player_{num}_x']
            a_p_Y_fh = synced_events.loc[:second_half_idx, f'player_{num}_y']
            a_p_X_sh = synced_events.loc[second_half_idx:, f'player_{num}_x']
            a_p_Y_sh = synced_events.loc[second_half_idx:, f'player_{num}_y']

            synced_events.loc[:second_half_idx, f'player_{num}_dist_to_goal'], \
            synced_events.loc[:second_half_idx, f'player_{num}_angle_to_goal'] = calc_angle_and_distance_to_goal(
                a_p_X_fh, a_p_Y_fh, False)
            synced_events.loc[second_half_idx:, f'player_{num}_dist_to_goal'], \
            synced_events.loc[second_half_idx:, f'player_{num}_angle_to_goal'] = calc_angle_and_distance_to_goal(
                a_p_X_sh, a_p_Y_sh, True)
        # away goalkeeper
        a_gk_X_fh = synced_events.loc[:second_half_idx, 'player_22_x']
        a_gk_Y_fh = synced_events.loc[:second_half_idx, 'player_22_y']
        a_gk_X_sh = synced_events.loc[second_half_idx:, 'player_22_x']
        a_gk_Y_sh = synced_events.loc[second_half_idx:, 'player_22_y']
        synced_events.loc[:second_half_idx, 'player_22_dist_to_goal'], \
        synced_events.loc[:second_half_idx, 'player_22_angle_to_goal'] = calc_angle_and_distance_to_goal(a_gk_X_fh,
                                                                                                         a_gk_Y_fh,
                                                                                                         False)
        synced_events.loc[second_half_idx:, 'player_22_dist_to_goal'], \
        synced_events.loc[second_half_idx:, 'player_22_angle_to_goal'] = calc_angle_and_distance_to_goal(a_gk_X_sh,
                                                                                                         a_gk_Y_sh,
                                                                                                         True)
        # home players
        for num in range(1, 11):
            h_p_X_fh = synced_events.loc[:second_half_idx, f'player_{num}_x']
            h_p_Y_fh = synced_events.loc[:second_half_idx, f'player_{num}_y']
            h_p_X_sh = synced_events.loc[second_half_idx:, f'player_{num}_x']
            h_p_Y_sh = synced_events.loc[second_half_idx:, f'player_{num}_y']

            synced_events.loc[:second_half_idx, f'player_{num}_dist_to_goal'], \
            synced_events.loc[:second_half_idx, f'player_{num}_angle_to_goal'] = calc_angle_and_distance_to_goal(
                h_p_X_fh, h_p_Y_fh, True)
            synced_events.loc[second_half_idx:, f'player_{num}_dist_to_goal'], \
            synced_events.loc[second_half_idx:, f'player_{num}_angle_to_goal'] = calc_angle_and_distance_to_goal(
                h_p_X_sh, h_p_Y_sh, False)
        # home goalkeeper - calc_angle_and_distance_to_goal(home player X, home player Y, True)
        h_gk_X_fh = synced_events.loc[:second_half_idx, 'player_21_x']
        h_gk_Y_fh = synced_events.loc[:second_half_idx, 'player_21_y']
        h_gk_X_sh = synced_events.loc[second_half_idx:, 'player_21_x']
        h_gk_Y_sh = synced_events.loc[second_half_idx:, 'player_21_y']
        synced_events.loc[:second_half_idx, 'player_21_dist_to_goal'], \
        synced_events.loc[:second_half_idx, 'player_21_angle_to_goal'] = calc_angle_and_distance_to_goal(h_gk_X_fh,
                                                                                                         h_gk_Y_fh,
                                                                                                         True)
        synced_events.loc[second_half_idx:, 'player_21_dist_to_goal'], \
        synced_events.loc[second_half_idx:, 'player_21_angle_to_goal'] = calc_angle_and_distance_to_goal(h_gk_X_sh,
                                                                                                         h_gk_Y_sh,
                                                                                                         False)

        # ball
        b_X_fh = synced_events.loc[:second_half_idx, 'ball_x']
        b_Y_fh = synced_events.loc[:second_half_idx, 'ball_y']
        b_X_sh = synced_events.loc[second_half_idx:, 'ball_x']
        b_Y_sh = synced_events.loc[second_half_idx:, 'ball_y']

        if event['event_team'] == 0:
            synced_events.loc[:second_half_idx, 'ball_dist_to_goal'], \
            synced_events.loc[:second_half_idx, 'ball_angle_to_goal'] = calc_angle_and_distance_to_goal(b_X_fh, b_Y_fh,
                                                                                                        True)
            synced_events.loc[second_half_idx:, 'ball_dist_to_goal'], \
            synced_events.loc[second_half_idx:, 'ball_angle_to_goal'] = calc_angle_and_distance_to_goal(b_X_sh, b_Y_sh,
                                                                                                        False)
            # calc_angle_and_distance_to_goal(ball X, ball Y, True)
        else:
            synced_events.loc[:second_half_idx, 'ball_dist_to_goal'], \
            synced_events.loc[:second_half_idx, 'ball_angle_to_goal'] = calc_angle_and_distance_to_goal(b_X_fh, b_Y_fh,
                                                                                                        False)
            synced_events.loc[second_half_idx:, 'ball_dist_to_goal'], \
            synced_events.loc[second_half_idx:, 'ball_angle_to_goal'] = calc_angle_and_distance_to_goal(b_X_sh, b_Y_sh,
                                                                                                        True)
            # calc_angle_and_distance_to_goal(ball X, ball Y, False)

    newids = []
    tl_x = []
    tl_y = []
    tl_speed = []
    tl_dir_x = []
    tl_dir_y = []
    tl_ang_tg = []
    tl_dist_tg = []
    tl_is_bc = []
    tl_dist_bc = []
    gl_x = []
    gl_y = []
    gl_speed = []
    gl_dir_x = []
    gl_dir_y = []
    gl_ang_tg = []
    gl_dist_tg = []
    gl_is_bc = []
    gl_dist_bc = []
    for ev_idx, event in synced_events.iterrows():
        # calc touchline and goalline features
        tl_fts = calc_touchline_features(event)
        tl_x.append(tl_fts[0])
        tl_y.append(tl_fts[1])
        tl_speed.append(tl_fts[2])
        tl_dir_x.append(tl_fts[3])
        tl_dir_y.append(tl_fts[4])
        tl_ang_tg.append(tl_fts[5])
        tl_dist_tg.append(tl_fts[6])
        tl_is_bc.append(tl_fts[7])
        tl_dist_bc.append(tl_fts[8])

        gl_fts = calc_goalline_features(event)
        gl_x.append(gl_fts[0])
        gl_y.append(gl_fts[1])
        gl_speed.append(gl_fts[2])
        gl_dir_x.append(gl_fts[3])
        gl_dir_y.append(gl_fts[4])
        gl_ang_tg.append(gl_fts[5])
        gl_dist_tg.append(gl_fts[6])
        gl_is_bc.append(gl_fts[7])
        gl_dist_bc.append(gl_fts[8])

        for num in range(1, 23):
            foundone=False
            if event[f'player_{num}_objectId'] == event['playerTrackingId']:
                newids.append(num)
                break
            elif num == 22 and not foundone:
                print(event)
    synced_events['p1_id'] = newids

    synced_events['tl_x'] = tl_x
    synced_events['tl_y'] = tl_y
    synced_events['tl_speed'] = tl_speed
    synced_events['tl_direction_x'] = tl_dir_x
    synced_events['tl_direction_y'] = tl_dir_y
    synced_events['tl_angle_to_goal'] = tl_ang_tg
    synced_events['tl_dist_to_goal'] = tl_dist_tg
    synced_events['tl_is_bc'] = tl_is_bc
    synced_events['tl_dist_to_bc'] = tl_dist_bc

    synced_events['gl_x'] = gl_x
    synced_events['gl_y'] = gl_y
    synced_events['gl_speed'] = gl_speed
    synced_events['gl_direction_x'] = gl_dir_x
    synced_events['gl_direction_y'] = gl_dir_y
    synced_events['gl_angle_to_goal'] = gl_ang_tg
    synced_events['gl_dist_to_goal'] = gl_dist_tg
    synced_events['gl_is_bc'] = gl_is_bc
    synced_events['gl_dist_to_bc'] = gl_dist_bc
    return synced_events


def calc_touchline_features(event):
    speed = dir_x = dir_y = is_bc = x = y = 0
    if event['typeId'] == 12:
        x = event['Pass_end_x']
        y = event['Pass_end_y']
    elif event['typeId'] == 5:
        x = event['x']
        y = event['y']
    ang_to_goal, dist_to_goal = calc_angle_and_distance_to_goal(x, y, False)
    ball_carrier = event['playerTrackingId']
    bc_x, bc_y = get_player_pos(ball_carrier, event)
    dist_to_bc = distance_pos(bc_x, bc_y, x, y)

    return [x, y, speed, dir_x, dir_y, ang_to_goal, dist_to_goal, is_bc, dist_to_bc]


def calc_goalline_features(event):
    speed = dir_x = dir_y = is_bc = ang_to_goal = dist_to_goal = 0
    x = GOAL_X
    y = GOAL_Y
    ball_carrier = event['playerTrackingId']
    bc_x, bc_y = get_player_pos(ball_carrier, event)
    dist_to_bc = distance_pos(bc_x, bc_y, x, y)

    return [x, y, speed, dir_x, dir_y, ang_to_goal, dist_to_goal, is_bc, dist_to_bc]


def get_closest_op(p1_num, frame):
    p1_x, p1_y = frame[f'player_{p1_num}_x'], frame[f'player_{p1_num}_y']
    min_dist = 1000
    cl_p = 0
    for num in range(1, 23):
        if frame[f'player_{p1_num}_teamId'] != frame[f'player_{num}_teamId']:
            plX, plY = frame[f'player_{num}_x'], frame[f'player_{num}_y']
            dist = distance_pos(p1_x, p1_y, plX, plY)
            if dist < min_dist:
                min_dist = dist
                cl_p = num

    return cl_p, min_dist


def get_pass_reciever(frame):
    pX, pY = frame['Pass_end_x'], frame['Pass_end_y']
    min_dist = 10000
    rec_tr_id = 0
    rec_send_dist = 0
    for num in range(1, 23):
        if frame['event_team'] == frame[f'player_{num}_teamId']:
            plX, plY = frame[f'player_{num}_x'], frame[f'player_{num}_y']
            dist = distance_pos(pX, pY, plX, plY)
            if dist < min_dist:
                min_dist = dist
                rec_tr_id = num  # int(event[f'player_{num}_objectId'])
                rec_send_dist = dist
    return rec_tr_id, rec_send_dist


def create_edge_features(synced_event_df):
    interactions = []
    labelCol1 = []
    OP_OP = []
    OP_DP = []
    OP_B = []
    OP_G = []
    DP_B = []
    DP_T = []
    B_T = []
    B_G = []
    # closest_player = ['tbd'] * len(synced_event_df.index)
    p1_ids = []
    p2_ids = []
    p2_dists = []

    # ide minden elágazásnál felvenni a 2 megfelelő (op/dp/b/t/g) oszlopot
    for ev_idx, event in synced_event_df.iterrows():
        p1_num = event['p1_id']
        p2_id = 0
        p2_dist = 0
        if ((event['typeId'] == 1) | (event['typeId'] == 2)):  # pass
            interactions.append('pass')
            p2_id, p2_dist = get_pass_reciever(event)
            labelCol1.append('OP_OP')
            OP_OP.append(1)
            OP_DP.append(0)
            OP_B.append(0)
            OP_G.append(0)
            DP_B.append(0)
            DP_T.append(0)
            B_T.append(0)
            B_G.append(0)
        elif ((event['typeId'] == 13) | (event['typeId'] == 14) | (event['typeId'] == 15)):
            interactions.append('shot')
            p2_id = 25
            p2_dist = event[f'player_{p1_num}_dist_to_goal']
            labelCol1.append('OP_G')
            OP_OP.append(0)
            OP_DP.append(0)
            OP_B.append(0)
            OP_G.append(1)
            DP_B.append(0)
            DP_T.append(0)
            B_T.append(0)
            B_G.append(0)
        elif (event['typeId'] == 3):
            interactions.append('dribble')
            p2_id = 23
            p2_dist = 0
            labelCol1.append('OP_B')
            OP_OP.append(0)
            OP_DP.append(0)
            OP_B.append(1)
            OP_G.append(0)
            DP_B.append(0)
            DP_T.append(0)
            B_T.append(0)
            B_G.append(0)
        elif (event['typeId'] == 8):
            interactions.append('interception')
            p2_id = 23
            p2_dist = 0
            labelCol1.append('DP_B')
            OP_OP.append(0)
            OP_DP.append(0)
            OP_B.append(0)
            OP_G.append(0)
            DP_B.append(1)
            DP_T.append(0)
            B_T.append(0)
            B_G.append(0)
        elif ((event['typeId'] == 7) | (event['typeId'] == 74)):
            interactions.append('tackle')
            p2_id, p2_dist = get_closest_op(p1_num, event)
            labelCol1.append('OP_DP')
            OP_OP.append(0)
            OP_DP.append(1)
            OP_B.append(0)
            OP_G.append(0)
            DP_B.append(0)
            DP_T.append(0)
            B_T.append(0)
            B_G.append(0)
        elif (event['typeId'] == 12):
            interactions.append('clearence')
            p2_id = 24
            p2_dist = distance_pos(event[f'player_{p1_num}_x'], event[f'player_{p1_num}_y'], \
                                   event['Pass_end_x'], event['Pass_end_y'])
            labelCol1.append('DP_T')
            OP_OP.append(0)
            OP_DP.append(0)
            OP_B.append(0)
            OP_G.append(0)
            DP_B.append(0)
            DP_T.append(1)
            B_T.append(0)
            B_G.append(0)
        elif (event['typeId'] == 5):
            interactions.append('ball out')
            p2_id = 24
            player_x, player_y = get_player_pos(event['playerTrackingId'], event)
            p2_dist = distance_pos(player_x, player_y, event['x'], event['y'])
            p1_num = 23
            labelCol1.append('B_T')
            OP_OP.append(0)
            OP_DP.append(0)
            OP_B.append(0)
            OP_G.append(0)
            DP_B.append(0)
            DP_T.append(0)
            B_T.append(1)
            B_G.append(0)
        elif (event['typeId'] == 16):
            interactions.append('goal')
            p2_id = 25
            p2_dist = event[f'player_{p1_num}_dist_to_goal']
            p1_num = 23
            labelCol1.append('B_G')
            OP_OP.append(0)
            OP_DP.append(0)
            OP_B.append(0)
            OP_G.append(0)
            DP_B.append(0)
            DP_T.append(0)
            B_T.append(0)
            B_G.append(1)

        p2_ids.append(p2_id)
        p2_dists.append(p2_dist)
        p1_ids.append(p1_num)

    return interactions, p1_ids, p2_ids, p2_dists, OP_OP, OP_DP, OP_B, OP_G, DP_B, DP_T, B_T, B_G


def create_structure(synced_event_df):
    structured_data = []
    for ev_idx, event in synced_event_df.iterrows():
        structured_event = []
        # player1
        p_tr_id = int(event['p1_id'])
        structured_event.append(p_tr_id)
        # player2
        r_tr_id = int(event['recipientId'])
        structured_event.append(r_tr_id)
        # timestamp
        timestamp = event['timestamp']
        structured_event.append(timestamp)
        # action
        action = event['typeId']
        structured_event.append(action)
        # edge features
        # structured_event.append(event['teammates'])
        structured_event.append(event['OP_OP'])
        structured_event.append(event['OP_DP'])
        structured_event.append(event['OP_B'])
        structured_event.append(event['OP_G'])
        structured_event.append(event['DP_B'])
        structured_event.append(event['DP_T'])
        structured_event.append(event['B_T'])
        structured_event.append(event['B_G'])
        structured_event.append(event['playerDistance'])
        # node features
        for num in range(1, 23):
            # player features
            structured_event.append(event[f'player_{num}_x'])
            structured_event.append(event[f'player_{num}_y'])
            structured_event.append(event[f'player_{num}_speed'])
            structured_event.append(event[f'player_{num}_direction_x'])
            structured_event.append(event[f'player_{num}_direction_y'])
            structured_event.append(event[f'player_{num}_angle_to_goal'])
            structured_event.append(event[f'player_{num}_dist_to_goal'])
            structured_event.append(event[f'player_{num}_is_bc'])
            structured_event.append(event[f'player_{num}_dist_to_bc'])
        # ball features
        structured_event.append(event['ball_x'])
        structured_event.append(event['ball_y'])
        structured_event.append(event['ball_speed'])
        structured_event.append(event['ball_direction_x'])
        structured_event.append(event['ball_direction_y'])
        structured_event.append(event['ball_angle_to_goal'])
        structured_event.append(event['ball_dist_to_goal'])
        structured_event.append(event['ball_is_bc'])
        structured_event.append(event['ball_dist_to_bc'])
        # touchline features
        structured_event.append(event['tl_x'])
        structured_event.append(event['tl_y'])
        structured_event.append(event['tl_speed'])
        structured_event.append(event['tl_direction_x'])
        structured_event.append(event['tl_direction_y'])
        structured_event.append(event['tl_angle_to_goal'])
        structured_event.append(event['tl_dist_to_goal'])
        structured_event.append(event['tl_is_bc'])
        structured_event.append(event['tl_dist_to_bc'])
        # goalline features
        structured_event.append(event['gl_x'])
        structured_event.append(event['gl_y'])
        structured_event.append(event['gl_speed'])
        structured_event.append(event['gl_direction_x'])
        structured_event.append(event['gl_direction_y'])
        structured_event.append(event['gl_angle_to_goal'])
        structured_event.append(event['gl_dist_to_goal'])
        structured_event.append(event['gl_is_bc'])
        structured_event.append(event['gl_dist_to_bc'])

        structured_data.append(structured_event)
    return pd.DataFrame(structured_data)


def create_dataset(rawpath, savepath, metadata_fn, init_match_i, final_match_i, ds_fn):
    data = []

    if final_match_i == 'last': final_match_i = len(available_matches)
    for i in range(init_match_i, final_match_i):
        print(f"match num {i + 1}, {available_matches[i]}")
        start = timer()
        match_id = available_matches[i]
        tracking, event = flatten_and_features(rawpath, match_id, metadata_fn)
        if tracking.empty or event.empty:
            print('problem: could not read the tracking file')
            continue
        is25fps = True
        if tracking.loc[1, 'timestamp'] - tracking.loc[0, 'timestamp'] == 100:
            is25fps = False
            print('switched to 10 fps.')
        # finalizing node features
        synced_events = sync_and_features(tracking, event, is25fps)

        # calulating edge features
        interactions, p1_ids, rec_tracking_ids, rec_send_dists, OP_OP, OP_DP, OP_B, OP_G, DP_B, DP_T, B_T, B_G = create_edge_features(
            synced_events)
        synced_events['typeId'] = interactions
        synced_events['p1_id'] = p1_ids
        synced_events['recipientId'] = rec_tracking_ids
        synced_events['playerDistance'] = rec_send_dists
        synced_events['OP_OP'] = OP_OP
        synced_events['OP_DP'] = OP_DP
        synced_events['OP_B'] = OP_B
        synced_events['OP_G'] = OP_G
        synced_events['DP_B'] = DP_B
        synced_events['DP_T'] = DP_T
        synced_events['B_T'] = B_T
        synced_events['B_G'] = B_G

        # synced_events.loc[((synced_events['typeId'] == 'pass') & synced_events['outcome'] == 0), 'recipientId'] = 0

        synced_events = create_structure(synced_events)
        # filling in nan values
        synced_events = synced_events.fillna(method='ffill')
        synced_events = synced_events.fillna(0)
        if synced_events.empty:
            print('problem: could not sync tracking and event data')
            continue
        for elem in synced_events.values.tolist():
            data.append(elem)
        # data.append(synced_events.values.tolist())
        print("with GPU:", timer() - start)
    # np.save(path+ds_fn,data)

    with open(savepath + ds_fn, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)
    print('saved ', ds_fn)


rawpath = os.getcwd() + '/raw_data/'
savepath = os.getcwd() + '/data/'

def get_available_matches(path):
    has_tracking = []
    has_event = []
    for f in os.listdir(path):
        if f.startswith('tracking-data') and f.endswith('25fps.txt'):
            has_tracking.append(f.split('-')[2])
        elif f.startswith('events-ma13'):
            has_event.append(f.split('-')[2][:-5])
    return (sorted(list(set(has_event).intersection(has_tracking))))


match_ovw = pd.read_csv(rawpath + '2021-10-21-stats-perform-data-availability (3).csv')
match_ovw = match_ovw.drop(columns=['competition_id', 'tournament_id', 'match_date', 'match_time', 'match_description',
    'match_week', 'match_status', 'ma1', 'ma2', 'ma3', 'tracking-10fps'])
available_matches = get_available_matches(rawpath)  # match_ovw[(match_ovw['ma13']==True) & (match_ovw['tracking-25fps']==True)]['match_id'].tolist()
metadataFileName = '2021-08-19-jpl-season-2020-2021-squads.csv'

print(len(available_matches))
create_dataset(rawpath, savepath, metadataFileName, 0, 'last', 'data.csv')
# create_dataset(path, metadataFileName,0,49,'data_1.csv')
# create_dataset(path, metadataFileName,50,99,'data_2.csv')
# create_dataset(path, metadataFileName,100,149,'data_3.csv')
# create_dataset(path, metadataFileName,150,199,'data_4.csv')
# create_dataset(path, metadataFileName,200,249,'data_5.csv')
# create_dataset(path, metadataFileName,250,'last','data_6.csv')
