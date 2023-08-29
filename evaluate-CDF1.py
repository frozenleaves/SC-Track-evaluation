import csv
import shutil

import pandas as pd

def handle_gt(gt_file):
    df = pd.read_csv(gt_file)
    result = set()
    gt_groups = df.groupby(['track_id', 'cell_id', 'frame_index', 'center_x', 'center_y', 'parent_id'])
    for name, gt_group in gt_groups:
        if name[1][-1] != '0':
            # print(name)
            result.add(name[1])
    return len(result)


def handle_sctrack(dataframe):
    df = pd.read_csv(dataframe)
    count = 0
    result = {}
    truth = {}
    for track_id, group in df.groupby("track_id"):

        if (len(set(group["cell_id"])) - 1) > 0 and (len(set(group["cell_id"])) - 1) % 2 == 0:
            # print(group['track_id'])
            cell_id_set = set(group['cell_id'])
            if f'{track_id}_1' in cell_id_set and f'{track_id}_2' in cell_id_set:
                truth[track_id] = cell_id_set
            result[track_id] = cell_id_set
            count += (len(cell_id_set) - 1)

    return count, result


def parse_pcnadeep(pcnadeep_file, gt_file):
    pred_df = pd.read_csv(pcnadeep_file)

    gt_df = pd.read_csv(gt_file)

    gt_groups = gt_df.groupby(['track_id', 'cell_id', 'frame_index', 'center_x', 'center_y', 'parent_id'])
    try:
        pred_groups = pred_df.groupby(['track_id', 'cell_id', 'frame_index', 'center_x', 'center_y', 'parent_id'])
    except KeyError:
        print(pcnadeep_file)

    cell_id_division = {}
    track_id_dict = {}
    track_id_position = {}
    for name, gt_group in gt_groups:
        # print(name)
        if name[0] not in track_id_dict:
            track_id_dict[name[0]] = {name[1]}
        else:
            track_id_dict[name[0]].add(name[1])
        if name[0] not in track_id_position:
            track_id_position[name[0]] = {name[1]: [(name[3], name[4])]}
        else:
            if name[1] not in track_id_position[name[0]]:
                track_id_position[name[0]][name[1]] = [(name[3], name[4])]
            else:
                track_id_position[name[0]][name[1]].append((name[3], name[4]))
    for i in track_id_dict:
        if len(track_id_dict[i]) > 0:
            cell_id_division[i] = track_id_dict[i]

    pred_track_id_position = {}
    pred_div = set()
    for name, pred_group in pred_groups:
        # print(name)
        if name[0] not in pred_track_id_position:
            pred_track_id_position[name[0]] = [(name[3], name[4])]
        else:
            pred_track_id_position[name[0]].append((name[3], name[4]))
        if name[5] != 0:
            pred_div.add(name[0])


    cell_id_map = {}
    for track_id in track_id_position:
        cell_id_dict = track_id_position[track_id]
        for cell_id in cell_id_dict:
            cell_id_position = cell_id_dict[cell_id]
            tmp = []
            for i in pred_track_id_position:
                pred_position = pred_track_id_position[i]
                for j in cell_id_position:
                    dst = [j[0] - pred_position[0][0], j[1] - pred_position[0][1]]
                    if abs(dst[0]) <1 and abs(dst[1]) < 1:
                        tmp.append(i)
                        break
            cell_id_map[cell_id] = tmp
        # break
    # print(cell_id_map)
    # print(cell_id_division)

    TP = 0
    FP = 0
    FN = 0
    GT = handle_gt(gt_file)
    for cell_id in cell_id_map:
        if cell_id[-1] != '0':
            # GT += 1
            for pred_id in cell_id_map[cell_id]:
                if pred_id in pred_div:
                    if len(cell_id_map[cell_id]) > 1:
                        TP += 1
                        FN += (len(cell_id_map[cell_id]) - 1)
                        break
                    else:
                        TP += 1
                else:
                    FN += 1
            if not cell_id_map[cell_id]:
                FN += 1
        else:
            for pred_id in cell_id_map[cell_id]:
                if pred_id in pred_div:
                    FP += 1
    if TP ==0:
        mdr = 0
    else:
        mdr = 2 * TP / (2 * TP + FP + FN)
    return GT, TP, FP, FN, mdr


def parse_trackmate(trackmate_file, gt_file, track_division_file):

    pred_df = pd.read_csv(trackmate_file)
    gt_df = pd.read_csv(gt_file)
    division_df = pd.read_csv(track_division_file)
    gt_groups = gt_df.groupby(['track_id', 'cell_id', 'frame_index', 'center_x', 'center_y', 'parent_id'])
    try:
        pred_groups = pred_df.groupby(['track_id', 'cell_id', 'frame_index', 'center_x', 'center_y'])
    except KeyError:
        return


    cell_id_division = {}
    track_id_dict = {}
    track_id_position = {}
    for name, gt_group in gt_groups:
        # print(name)
        if name[0] not in track_id_dict:
            track_id_dict[name[0]] = {name[1]}
        else:
            track_id_dict[name[0]].add(name[1])
        if name[0] not in track_id_position:
            track_id_position[name[0]] = {name[1]: [(name[3], name[4])]}
        else:
            if name[1] not in track_id_position[name[0]]:
                track_id_position[name[0]][name[1]] = [(name[3], name[4])]
            else:
                track_id_position[name[0]][name[1]].append((name[3], name[4]))
    for i in track_id_dict:
        if len(track_id_dict[i]) > 0:
            cell_id_division[i] = track_id_dict[i]

    pred_track_id_position = {}
    pred_div = set()
    division_map = {}
    for name, pred_group in pred_groups:
        # print(name)
        if name[0] not in pred_track_id_position:
            pred_track_id_position[name[0]] = [(name[3], name[4])]
        else:
            pred_track_id_position[name[0]].append((name[3], name[4]))
        result = division_df[division_df['track_id'] == name[0]]
        if result['division_events'].values[0] != 0:
            pred_div.add(name[0])
            division_map[name[0]] = result['division_events'].values[0]

        # if name[5] != 0:
        #     pred_div.add(name[0])

    cell_id_map = {}
    for track_id in track_id_position:
        cell_id_dict = track_id_position[track_id]
        for cell_id in cell_id_dict:
            cell_id_position = cell_id_dict[cell_id]
            tmp = []
            for i in pred_track_id_position:
                pred_position = pred_track_id_position[i]
                for j in cell_id_position:
                    dst = [j[0] - pred_position[0][0], j[1] - pred_position[0][1]]
                    if abs(dst[0]) <2 and abs(dst[1]) < 2:
                        tmp.append(i)
                        break
            cell_id_map[cell_id] = tmp
    TP = 0
    FP = 0
    FN = 0
    GT = handle_gt(gt_file)
    # print(cell_id_map)
    for cell_id in cell_id_map:
        # print(cell_id)
        handle_flag = False
        if cell_id[-1] != '0':
            # GT += 1
            for pred_id in cell_id_map[cell_id]:
                if not handle_flag:
                    if pred_id in pred_div:
                        division_events = division_map[pred_id]
                        if division_events > 2:
                            TP += 2
                            FN += (division_events - 2)
                            break
                        else:
                            TP += 2
                    handle_flag = True
                else:
                    if pred_id in pred_div:
                        division_events = division_map[pred_id]
                        FP += division_events
        else:
            for pred_id in cell_id_map[cell_id]:
                if pred_id in pred_div:
                    FP += division_map[pred_id]
    FN = GT - TP
    if TP == 0:
        mdr = 0
    else:
        mdr = 2 * TP / (2 * TP + FP + FN)
    return GT, TP, FP, FN, mdr


def parse_sctrack(sctrack_file, gt_file):
    gt = handle_gt(gt_file)
    predict = handle_sctrack(sctrack_file)
    f = gt - predict[0]
    TP = gt if predict[0] > gt else predict[0]
    if f >= 0:
        FN = f
        FP = 0
    else:
        FP = abs(f)
        FN = 0
    mdr = 2 * TP / (2 * TP + FP + FN)
    return TP, FP, FN, mdr


def test():
    with open(fr'G:\paper\evaluate_data\evaluate_for_tracking\MDR(base)\5-base-MDR-new.csv', 'w',
              newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video ID', 'GT', 'sctrack_TP', 'sctrack_FP', 'sctrack_FN', 'sctrack_CDF1',
                         'GT_p', 'pcnadeep_TP', 'pcnadeep_FP', 'pcnadeep_FN', 'pcnadeep_CDF1',
                         'GT_t', 'trackmate_TP', 'trackmate_FP', 'trackmate_FN', 'trackmate_CDF1'])
        prediction_sctrack = rf'G:\paper\evaluate_data\5min\copy_of_1_xy01_5min\tracking_output\track.csv'
        prediction_pcnadeep = rf'G:\paper\evaluate_data\5min\copy_of_1_xy01_5min\track\refined-pcnadeep(CCDeep_format).csv'
        prediction_GT = rf"G:\paper\evaluate_data\5min\copy_of_1_xy01_5min\5-track-GT.csv"
        prediction_trackmeta = rf"G:\paper\evaluate_data\5min\copy_of_1_xy01_5min\trackmate-new.csv"
        prediction_trackmeta_div = rf"G:\paper\evaluate_data\5min\copy_of_1_xy01_5min\trackmate-division.csv"

        sctrack_ret = parse_sctrack(prediction_sctrack, prediction_GT)
        pcnadeep_ret = parse_pcnadeep(prediction_pcnadeep, prediction_GT)
        trackmate_ret = parse_trackmate(prediction_trackmeta, prediction_GT, prediction_trackmeta_div)
        GT = handle_gt(prediction_GT)
        rows = ['copy_of_1_xy01', GT, *sctrack_ret, *pcnadeep_ret, *trackmate_ret]
        writer.writerow(rows)