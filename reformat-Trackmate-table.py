# -*- coding: utf-8 -*-

# Modify the Trackmate output result to make its format compatible with sctrack

import pandas as pd
import os


def handle(input, output):
    """Reformat tracking results"""
    if not os.path.exists(input):
        return
    df = pd.read_csv(input, skiprows=3)

    name = 'LABEL	ID	TRACK_ID	QUALITY	POSITION_X	POSITION_Y	POSITION_Z	POSITION_T	FRAME	RADIUS	VISIBILITY	MANUAL_SPOT_COLOR	MEAN_INTENSITY_CH1	MEDIAN_INTENSITY_CH1	MIN_INTENSITY_CH1	MAX_INTENSITY_CH1	TOTAL_INTENSITY_CH1	STD_INTENSITY_CH1	CONTRAST_CH1	SNR_CH1'

    name = name.split('\t')

    rename_map = dict(zip(list(df.columns), name))

    df = df.rename(columns=rename_map)

    # 提取需要的列数据
    try:
        df_new = df[['TRACK_ID', 'POSITION_X', 'POSITION_Y', 'FRAME']]
    except KeyError:
        new_name = 'frame_index	track_id	cell_id	center_x	center_y'.split('\t')
        df = df.rename(columns=dict(zip(list(df.columns), new_name)))
        df.to_csv(output, index=False)
        return

    df_new.insert(loc=df_new.columns.get_loc('TRACK_ID') + 1, column='cell_id', value=df_new['TRACK_ID'])

    df_new = df_new.sort_values(['TRACK_ID', 'cell_id', 'FRAME'])

    cols = list(df_new.columns)
    cols.insert(0, cols.pop(cols.index('FRAME')))
    df_new = df_new.reindex(columns=cols)
    new_name = 'frame_index	track_id	cell_id	center_x	center_y'.split('\t')
    df_new = df_new.rename(columns=dict(zip(list(df_new.columns), new_name)))

    print(df_new)

    df_new.to_csv(output, index=False)


def handle_division(track_file):
    """Reformat split statistics results"""
    df = pd.read_csv(track_file, skiprows=3, encoding=u'gbk', index_col=0)

    name = r'LABEL  TRACK_INDEX	TRACK_ID	NUMBER_SPOTS	NUMBER_GAPS	NUMBER_SPLITS	NUMBER_MERGES	NUMBER_COMPLEX	LONGEST_GAP	TRACK_DURATION	TRACK_START	TRACK_STOP	TRACK_DISPLACEMENT	TRACK_X_LOCATION	TRACK_Y_LOCATION	TRACK_Z_LOCATION	TRACK_MEAN_SPEED	TRACK_MAX_SPEED	TRACK_MIN_SPEED	TRACK_MEDIAN_SPEED	TRACK_STD_SPEED	TRACK_MEAN_QUALITY	TOTAL_DISTANCE_TRAVELED	MAX_DISTANCE_TRAVELED	CONFINEMENT_RATIO	MEAN_STRAIGHT_LINE_SPEED	LINEARITY_OF_FORWARD_PROGRESSION	MEAN_DIRECTIONAL_CHANGE_RATE'
    name = name.split('\t')
    rename_map = dict(zip(list(df.columns), name))
    df = df.rename(columns=rename_map)
    print(df[df.columns[:]])
    df_new = df[['TRACK_ID', 'NUMBER_SPLITS']]
    df_new.insert(loc=df_new.columns.get_loc('TRACK_ID') + 1, column='cell_id', value=df_new['TRACK_ID'])

    df_new = df_new.sort_values(['TRACK_ID', 'cell_id'])
    print(df_new)
    new_name = ['track_id', 'cell_id', 'division_events']
    df_new = df_new.rename(columns=dict(zip(list(df_new.columns), new_name)))
    df_new.to_csv(fr'{os.path.dirname(track_file)}\trackmate-division.csv', index=False)


def test():
    dirs = ['copy_of_1_xy01', 'copy_of_1_xy19', 'MCF10A_copy02', 'MCF10A_copy11', 'src06']
    base = r"G:\paper\evaluate_data\5min"
    for i in dirs:
        input = rf"{base}\{i}_5min\export.csv"
        input_div = rf"{base}\{i}_5min\export-track.csv"
        output = rf"{base}\{i}_5min\trackmate-new.csv"
        handle(input, output)
        handle_division(input_div)
