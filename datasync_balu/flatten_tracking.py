import pandas as pd

tracking_cols=['timestamp','time_of_current_half','half_indicator','match_not_in_play',
               'player_1_teamId', 'player_1_objectId','player_1_jerseyNum','player_1_x','player_1_y',
               'player_2_teamId', 'player_2_objectId','player_2_jerseyNum','player_2_x','player_2_y',
               'player_3_teamId', 'player_3_objectId','player_3_jerseyNum','player_3_x','player_3_y',
               'player_4_teamId', 'player_4_objectId','player_4_jerseyNum','player_4_x','player_4_y',
               'player_5_teamId', 'player_5_objectId','player_5_jerseyNum','player_5_x','player_5_y',
               'player_6_teamId', 'player_6_objectId','player_6_jerseyNum','player_6_x','player_6_y',
               'player_7_teamId', 'player_7_objectId','player_7_jerseyNum','player_7_x','player_7_y',
               'player_8_teamId', 'player_8_objectId','player_8_jerseyNum','player_8_x','player_8_y',
               'player_9_teamId', 'player_9_objectId','player_9_jerseyNum','player_9_x','player_9_y',
               'player_10_teamId', 'player_10_objectId','player_10_jerseyNum','player_10_x','player_10_y',
               'player_11_teamId', 'player_11_objectId','player_11_jerseyNum','player_11_x','player_11_y',
               'player_12_teamId', 'player_12_objectId','player_12_jerseyNum','player_12_x','player_12_y',
               'player_13_teamId', 'player_13_objectId','player_13_jerseyNum','player_13_x','player_13_y',
               'player_14_teamId', 'player_14_objectId','player_14_jerseyNum','player_14_x','player_14_y',
               'player_15_teamId', 'player_15_objectId','player_15_jerseyNum','player_15_x','player_15_y',
               'player_16_teamId', 'player_16_objectId','player_16_jerseyNum','player_16_x','player_16_y',
               'player_17_teamId', 'player_17_objectId','player_17_jerseyNum','player_17_x','player_17_y',
               'player_18_teamId', 'player_18_objectId','player_18_jerseyNum','player_18_x','player_18_y',
               'player_19_teamId', 'player_19_objectId','player_19_jerseyNum','player_19_x','player_19_y',
               'player_20_teamId', 'player_20_objectId','player_20_jerseyNum','player_20_x','player_20_y',
               'player_21_teamId', 'player_21_objectId','player_21_jerseyNum','player_21_x','player_21_y',
               'player_22_teamId', 'player_22_objectId','player_22_jerseyNum','player_22_x','player_22_y',
               'to_del_1',
               'ball_x','ball_y','ball_z',
               'to_del_2']
filename='tracking-data-7ky4x0axer75pyu4yskq0ynai-25fps.txt'
path = 'C:\\Users\\mibam\\egyetem\\sports_analytics\\BEL_data\\'
tracking = pd.read_csv(path+filename,sep=";|,|:",names=tracking_cols, header=None,engine='python')
tracking=tracking.drop(labels=['to_del_1','to_del_2'], axis=1)

temp = tracking[tracking['half_indicator']==1]['time_of_current_half'].max()
tracking['timeMilliSec'] = tracking['time_of_current_half']+temp*(tracking['half_indicator']-1)

tracking.to_csv(path+'flat-tracking-7ky4x0axer75pyu4yskq0ynai-25fps.csv',index=False)