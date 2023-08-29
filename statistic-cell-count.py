import csv
import json
import math
import os

import numpy as np
import pandas as pd
import tifffile

def total(file):
    count = 0
    if file.endswith('.json'):
        with open(file) as f:
            data = json.load(f)
            for i in data:
                regions = data[i]['regions']
                count += len(regions)
    elif file.endswith('.tif'):
        image = tifffile.imread(file)
        for i in image:
            num = np.unique(i)
            count += len(num)
    return count

def avg(file):
    if file.endswith('.json'):
        with open(file) as f:
            data = json.load(f)
            frame = len(data)
            count = 0
            for i in data:
                regions = data[i]['regions']
                count += len(regions)
        return round(count / frame, 4)
    elif file.endswith('.tif'):
        count = 0
        image = tifffile.imread(file)
        frame = image.shape[0]
        for i in image:
            num = np.unique(i)
            count += len(num)
        return round(count / frame, 4)


def each_phase_count(file):
    """Count the number of cells in each cycle, with file being the tracked ground truth data"""
    df = pd.read_csv(file)
    phase_col = df['phase']
    # print(phase_col.value_counts())
    value = phase_col.value_counts().to_numpy()
    print([os.path.dirname(file)] +  [i for i in value])


def noise_phase(file):
    """Statistical noise classification results"""
    with open(file) as f:
        data = json.load(f)
        G = 0
        S = 0
        M = 0
        for i in data:
            regions = data[i]['regions']
            for r in regions:
                p = r["region_attributes"]["phase"]
                if p == 'G1/G2':
                    G += 1
                if p == 'S':
                    S += 1
                if p == 'M':
                    M += 1
    return f'{G}, {S}, {M}'


def compare(gt_phase, predict_phase):
    """
    Compare the records of GT and predict, and if they match, return the corresponding GT value.
    Otherwise, return False based on the error type
    """
    # Fs = FP + FN
    record = {'G': False, 'S': False, 'M': False, 'FP_G': False, 'FP_S': False,
              'FP_M': False,
              'FN_G': False, 'FN_S': False, 'FN_M': False
              }

    if predict_phase == gt_phase:
        record[predict_phase[0]] = True
    else:
        record['FP_' + predict_phase[0]] = True
        record['FN_' + gt_phase[0]] = True
    return record

def compare_ignore_g1_g2(gt_phase, predict_phase):
    record = {'G': False, 'S': False, 'M': False, 'FP_G': False, 'FP_S': False,
              'FP_M': False,
              'FN_G': False, 'FN_S': False, 'FN_M': False
              }

    if predict_phase[0] == gt_phase[0]:
        record[predict_phase[0]] = True
    else:
        record['FP_' + predict_phase[0]] = True
        record['FN_' + gt_phase[0]] = True
    return record

def classify_detail(gt, pred):
    TP_G = 0
    TP_S = 0
    TP_M = 0
    FP_G = 0
    FP_S = 0
    FP_M = 0
    FN_G = 0
    FN_S = 0
    FN_M = 0
    
    with open(gt) as f:
        data_gt = json.load(f)
    with open(pred) as f2:
        data_pred = json.load(f2)

    keys = list(data_gt.keys())
    for k in keys:
        regions_gt = data_gt[k]['regions']
        regions_pred = data_pred[k]['regions']
        for g, p in zip(regions_gt, regions_pred):
            assert g['shape_attributes'] == p['shape_attributes']
            gt_phase = g["region_attributes"]["phase"]
            pred_phase = p["region_attributes"]["phase"]
            match_result = compare(gt_phase, pred_phase)
            # CCDeep_NUM += 1
            if match_result['G']:
                TP_G += 1
            if match_result['S']:
                TP_S += 1
            if match_result['M']:
                TP_M += 1
            if match_result['FP_G']:
                FP_G += 1
            if match_result['FP_S']:
                FP_S += 1
            if match_result['FP_M']:
                FP_M += 1
            if match_result['FN_G']:
                FN_G += 1
            if match_result['FN_S']:
                FN_S += 1
            if match_result['FN_M']:
                FN_M += 1
    return f'{TP_G}, {TP_S}, {TP_M}, {FP_G}, {FP_S}, {FP_M}, {FN_G}, {FN_S}, {FN_M}'

        # break



def time_resolution():
    dirs = ['copy_of_1_xy01', 'copy_of_1_xy19', 'MCF10A_copy02', 'MCF10A_copy11', 'src06']
    # dirs = ['copy_of_1_xy01', 'copy_of_1_xy19', 'src06']
    base = r'G:\paper\evaluate_data'
    for i in dirs:
        print(i)
        for gap in range(1, 5):
            ann = rf'G:\paper\evaluate_data\{gap*5}min\{i}_{gap*5}min\{gap*5}-result-GT.json'
            print(avg(ann))


def loss_detection():
    dirs = ['copy_of_1_xy01', 'copy_of_1_xy19', 'MCF10A_copy02', 'MCF10A_copy11', 'src06']
    gap = 1
    for i in dirs:
        print(i)
        ratio = [0.05, 0.1, 0.2, 0.3, 0.5]
        for r in ratio:
            ann = rf'G:\paper\evaluate_data\{gap * 5}min\{i}_{gap * 5}min\detection_loss_test\{int(r * 100)}%\{int(r * 100)}%-{gap * 5}-result-GT.json'
            print(avg(ann))


def noise_classify():
    base = r'G:\paper\evaluate_data\evaluate_noise_classification'
    for i in os.listdir(base):
        ann = os.path.join(base, f'{i}\\group-{str(i[-1])*4}.json')

        print(avg(ann))


def center(x_coords, y_coords):
    n = len(x_coords)
    area = 0.0
    centroid_x = 0.0
    centroid_y = 0.0
    for i in range(n):
        j = (i + 1) % n
        cross_product = x_coords[i] * y_coords[j] - x_coords[j] * y_coords[i]
        area += cross_product
        centroid_x += (x_coords[i] + x_coords[j]) * cross_product
        centroid_y += (y_coords[i] + y_coords[j]) * cross_product
    area /= 2.0
    centroid_x /= 6.0 * area
    centroid_y /= 6.0 * area
    return centroid_x, centroid_y

def euclidean_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def evaluate_detect(gt, pred):
    """evaluate object detection"""

    TP_G = 0
    TP_S = 0
    TP_M = 0
    FP_G = 0
    FP_S = 0
    FP_M = 0
    FN_G = 0
    FN_S = 0
    FN_M = 0

    with open(gt) as f:
        data_gt = json.load(f)
    with open(pred) as f2:
        data_pred = json.load(f2)

    keys = list(data_gt.keys())
    tp = 0
    fp = 0
    fn = 0
    for k in keys:

        regions_gt = data_gt[k]['regions']
        k = k.replace('copy_of_1_xy01', 'mcy')
        k = k.replace('new-', '')
        regions_pred = data_pred[k]['regions']
        # print('gt: ', len(regions_gt))
        # print('pred: ', len(regions_pred))

        unmatched_pred = []
        for rp in regions_pred:
            try:
                center_pred = center(rp['shape_attributes']["all_points_x"], rp['shape_attributes']["all_points_y"])
                phase_pred = rp["region_attributes"]["phase"]
            except KeyError:
                continue
            matched_flag = False
            matched_gt = None
            for rg in regions_gt:
                try:
                    center_gt = center(rg['shape_attributes']["all_points_x"], rg['shape_attributes']["all_points_y"])
                    phase_gt = rg["region_attributes"]["phase"]
                except KeyError:
                    continue
                if euclidean_distance(center_pred, center_gt) < 15:
                    # print('gt: ', center_gt)
                    # print('pred: ', center_pred)
                    matched_flag = True
                    matched_gt = rg
                    tp += 1
                    match_result = compare(phase_gt, phase_pred)
                    # CCDeep_NUM += 1
                    if match_result['G']:
                        TP_G += 1
                    if match_result['S']:
                        TP_S += 1
                    if match_result['M']:
                        TP_M += 1
                    if match_result['FP_G']:
                        FP_G += 1
                    if match_result['FP_S']:
                        FP_S += 1
                    if match_result['FP_M']:
                        FP_M += 1
                    if match_result['FN_G']:
                        FN_G += 1
                    if match_result['FN_S']:
                        FN_S += 1
                    if match_result['FN_M']:
                        FN_M += 1
                    break
            if not matched_flag:
                unmatched_pred.append(rp)
            else:
                regions_gt.remove(matched_gt)


        # print('unmatched pred: ', len(unmatched_pred))
        # print('unmatched gt: ', len(regions_gt))
        for unmatched_pred_region in unmatched_pred:
            if unmatched_pred_region["region_attributes"]["phase"] == 'G1/G2':
                FP_G += 1
            if unmatched_pred_region["region_attributes"]["phase"] == 'S':
                FP_S += 1
            if unmatched_pred_region["region_attributes"]["phase"] == 'M':
                FP_M += 1
            fp += 1

        for unmatched_gt_region in regions_gt:
            if unmatched_gt_region["region_attributes"]["phase"] == 'G1/G2':
                FN_G += 1
            if unmatched_gt_region["region_attributes"]["phase"] == 'S':
                FN_S += 1
            if unmatched_gt_region["region_attributes"]["phase"] == 'M':
                FN_M += 1
            fn += 1
        # print('===' * 10)
    # print(tp, fp, fn)
    ret = f'{tp}, {fp}, {fn}, {TP_G}, {TP_S}, {TP_M}, {FP_G}, {FP_S}, {FP_M}, {FN_G}, {FN_S}, {FN_M}'
    return ret



def smooth_evaluate(gt, pred):

    TP_G = 0
    TP_S = 0
    TP_M = 0
    FP_G = 0
    FP_S = 0
    FP_M = 0
    FN_G = 0
    FN_S = 0
    FN_M = 0

    with open(gt) as f:
        data_gt = json.load(f)
    with open(pred) as f2:
        data_pred = json.load(f2)

    keys = list(data_gt.keys())

    for k in keys:
        regions_gt = data_gt[k]['regions']
        try:
            regions_pred = data_pred[k]['regions']
        except KeyError:
            import re
            try:
                kn = re.sub('.*-0', 'mcy-0', k)
                regions_pred = data_pred[kn]['regions']
            except KeyError:
                kn = re.sub('.*-0', 'mcy-sub1-new-0', k)
                regions_pred = data_pred[kn]['regions']
        # print('gt: ', len(regions_gt))
        # print('pred: ', len(regions_pred))

        unmatched_pred = []
        for rp in regions_pred:
            try:
                center_pred = center(rp['shape_attributes']["all_points_x"], rp['shape_attributes']["all_points_y"])
            except KeyError:
                continue
            phase_pred = rp["region_attributes"]["phase"]
            matched_flag = False
            matched_gt = None
            for rg in regions_gt:
                try:
                    center_gt = center(rg['shape_attributes']["all_points_x"], rg['shape_attributes']["all_points_y"])
                except KeyError:
                    continue
                phase_gt = rg["region_attributes"]["phase"]
                if euclidean_distance(center_pred, center_gt) < 15:
                    # print('gt: ', center_gt)
                    # print('pred: ', center_pred)
                    matched_flag = True
                    matched_gt = rg
                    match_result = compare_ignore_g1_g2(phase_gt, phase_pred)
                    # CCDeep_NUM += 1
                    if match_result['G']:
                        TP_G += 1
                    if match_result['S']:
                        TP_S += 1
                    if match_result['M']:
                        TP_M += 1
                    if match_result['FP_G']:
                        FP_G += 1
                    if match_result['FP_S']:
                        FP_S += 1
                    if match_result['FP_M']:
                        FP_M += 1
                    if match_result['FN_G']:
                        FN_G += 1
                    if match_result['FN_S']:
                        FN_S += 1
                    if match_result['FN_M']:
                        FN_M += 1
                    break
            if not matched_flag:
                unmatched_pred.append(rp)
            else:
                regions_gt.remove(matched_gt)


        # print('unmatched pred: ', len(unmatched_pred))
        # print('unmatched gt: ', len(regions_gt))
        for unmatched_pred_region in unmatched_pred:
            if unmatched_pred_region["region_attributes"]["phase"][0] == 'G':
                FP_G += 1
            if unmatched_pred_region["region_attributes"]["phase"] == 'S':
                FP_S += 1
            if unmatched_pred_region["region_attributes"]["phase"] == 'M':
                FP_M += 1

        for unmatched_gt_region in regions_gt:
            if unmatched_gt_region["region_attributes"]["phase"][0] == 'G':
                FN_G += 1
            if unmatched_gt_region["region_attributes"]["phase"] == 'S':
                FN_S += 1
            if unmatched_gt_region["region_attributes"]["phase"] == 'M':
                FN_M += 1
    F1_G = (2 * TP_G) / (2 * TP_G + FP_G + FN_G)
    F1_S = (2 * TP_S) / (2 * TP_S + FP_S + FN_S)
    F1_M = (2 * TP_M) / (2 * TP_M + FP_M + FN_M)
    return f'{TP_G}, {TP_S}, {TP_M}, {FP_G}, {FP_S}, {FP_M}, {FN_G}, {FN_S}, {FN_M}, {F1_G}, {F1_S}, {F1_M}'



def loss_detection_cell_count(gt, pred):

    TP_G = 0
    TP_S = 0
    TP_M = 0
    FP_G = 0
    FP_S = 0
    FP_M = 0
    FN_G = 0
    FN_S = 0
    FN_M = 0

    with open(gt) as f:
        data_gt = json.load(f)
    with open(pred) as f2:
        data_pred = json.load(f2)

    keys = list(data_gt.keys())

    for k in keys:
        regions_gt = data_gt[k]['regions']
        try:
            regions_pred = data_pred[k]['regions']
        except KeyError:
            regions_pred = data_pred[k.replace('mcy', 'rpe19')]['regions']
        # print('gt: ', len(regions_gt))
        # print('pred: ', len(regions_pred))

        unmatched_pred = []
        for rp in regions_pred:
            try:
                center_pred = center(rp['shape_attributes']["all_points_x"], rp['shape_attributes']["all_points_y"])
                phase_pred = rp["region_attributes"]["phase"]
            except KeyError:
                continue
            matched_flag = False
            matched_gt = None
            for rg in regions_gt:
                try:
                    center_gt = center(rg['shape_attributes']["all_points_x"], rg['shape_attributes']["all_points_y"])
                    phase_gt = rg["region_attributes"]["phase"]
                except KeyError:
                    continue
                if euclidean_distance(center_pred, center_gt) < 15:
                    # print('gt: ', center_gt)
                    # print('pred: ', center_pred)
                    matched_flag = True
                    matched_gt = rg
                    match_result = compare_ignore_g1_g2(phase_gt, phase_pred)
                    # CCDeep_NUM += 1
                    if match_result['G']:
                        TP_G += 1
                    if match_result['S']:
                        TP_S += 1
                    if match_result['M']:
                        TP_M += 1
                    if match_result['FP_G']:
                        FP_G += 1
                    if match_result['FP_S']:
                        FP_S += 1
                    if match_result['FP_M']:
                        FP_M += 1
                    if match_result['FN_G']:
                        FN_G += 1
                    if match_result['FN_S']:
                        FN_S += 1
                    if match_result['FN_M']:
                        FN_M += 1
                    break
            if not matched_flag:
                unmatched_pred.append(rp)
            else:
                regions_gt.remove(matched_gt)


        # print('unmatched pred: ', len(unmatched_pred))
        # print('unmatched gt: ', len(regions_gt))
        for unmatched_pred_region in unmatched_pred:
            if unmatched_pred_region["region_attributes"]["phase"][0] == 'G':
                FP_G += 1
            if unmatched_pred_region["region_attributes"]["phase"] == 'S':
                FP_S += 1
            if unmatched_pred_region["region_attributes"]["phase"] == 'M':
                FP_M += 1

        for unmatched_gt_region in regions_gt:
            if unmatched_gt_region["region_attributes"]["phase"][0] == 'G':
                FN_G += 1
            if unmatched_gt_region["region_attributes"]["phase"] == 'S':
                FN_S += 1
            if unmatched_gt_region["region_attributes"]["phase"] == 'M':
                FN_M += 1
    F1_G = (2 * TP_G) / (2 * TP_G + FP_G + FN_G)
    F1_S = (2 * TP_S) / (2 * TP_S + FP_S + FN_S)
    F1_M = (2 * TP_M) / (2 * TP_M + FP_M + FN_M)
    return f'{TP_G}, {TP_S}, {TP_M}, {FP_G}, {FP_S}, {FP_M}, {FN_G}, {FN_S}, {FN_M}, {F1_G}, {F1_S}, {F1_M}'


def run_loss_detection():
    gap = 1
    dirs = ['copy_of_1_xy01', 'copy_of_1_xy19', 'MCF10A_copy02', 'MCF10A_copy11', 'src06']
    for i in dirs:
        output = rf'G:\paper\evaluate_data\{gap * 5}min\{i}_{gap * 5}min\detection_loss_test\evaluate_detection_loss.csv'
        # with open(output, 'w', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(
        #         ['tracker', 'detection loss ratio', 'Cell_NUM', 'TP_G1', 'TP_S', 'TP_G2', 'TP_M', 'FP_G1', 'FP_S',
        #          'FP_G2',
        #          'FP_M', 'FN_G1', 'FN_S', 'FN_G2', 'FN_M',
        #          'recall_G1', 'recall_S', 'recall_G2', 'recall_M', 'recall',
        #          'precision_G1', 'precision_S', 'precision_G2', 'precision_M', 'precision',
        #          'F1_G1', 'F1_S', 'F1_G2', 'F1_M', 'F1'])
        ratio = [0.05, 0.1, 0.2, 0.3, 0.5]
        for r in ratio:
            if r == 0:
                prediction = rf'G:\paper\evaluate_data\{gap * 5}min\{i}_{gap * 5}min\{gap*5}-result-GT.json'

                GT = rf"G:\paper\evaluate_data\{gap * 5}min\{i}_{gap * 5}min\{gap*5}-result-GT.json"
            else:
                prediction = rf'G:\paper\evaluate_data\{gap * 5}min\{i}_{gap * 5}min\detection_loss_test\{int(r * 100)}%\{int(r * 100)}%-{gap*5}-result-GT.json'
                GT = rf"G:\paper\evaluate_data\{gap * 5}min\{i}_{gap * 5}min\{gap*5}-result-GT.json"
            ret = loss_detection_cell_count(GT, prediction)
            print(ret)


def evaluate_raw_segment():
    base_pred = r'G:\paper\evaluate_data\incorrect-data-test\normal-dataset'
    base_gt = r'G:\paper\evaluate_data\5min'
    dirs = ['copy_of_1_xy01', 'copy_of_1_xy19', 'MCF10A_copy02', 'MCF10A_copy11', 'src06']
    for i in dirs:
        gt = fr"{base_gt}\{i}_5min\5-result-GT.json"
        pred = rf"{base_pred}\{i}\result.json"
        ret = evaluate_detect(gt, pred)
        print(ret)


def run_smooth():
    path = r'G:\paper\evaluate_data\evaluate_smoothing'
    dirs = ['copy_of_1_xy01', 'copy_of_1_xy19', 'MCF10A_copy02', 'MCF10A_copy11', 'src06']
    # dirs = ['copy_of_1_xy19']
    for i in dirs:
        gt = f'{path}\\{i}\\5-result-GT.json'
        pred_raw = f'{path}\\{i}\\10%-GT.json'
        pred_raw = f'{path}\\{i}\\result.json'
        pred_smooth = f'{path}\\{i}\\tracking_output\\result_with_track.json'
        # pred_smooth = f'G:\\paper\\evaluate_data\\incorrect-data-test\\normal-dataset\\{i}\\tracking_output\\result_with_track.json'
        ret_raw = smooth_evaluate(gt, pred_raw)
        ret_smooth = smooth_evaluate(gt, pred_smooth)
        # ret_raw = evaluate_detect(gt, pred_raw)
        # ret_smooth = evaluate_detect(gt, pred_smooth)
        print(f'{i}-raw, {ret_raw}')
        print(f'{i}-smooth, {ret_smooth}')

if __name__ == '__main__':
    # for i in range(1, 9):
    #     file = rf"G:\paper\test\Data{i}\SEG.tif"
    #     t = total(file)
    #     a = avg(file)
    #     print(t, ',',a)

    # path = r'G:\paper\evaluate_data\5min'
    # for i in os.listdir(path):
    #     file = os.path.join(path, f'{i}\\5-result-GT.json')
    #     t = total(file)
    #     a = avg(file)
    #     print(t, ',',a)

    # time_resolution()
    # loss_detection()
    # noise_classify()
    # each_phase_count(r"G:\paper\evaluate_data\evaluate_noise_classification\group0\track-GT.csv")

    # path = r'G:\paper\evaluate_data\5min'
    # for i in os.listdir(path):
    #     file = os.path.join(path, f'{i}\\5-track-GT.csv')
    #     each_phase_count(file)

    # path = r'G:\paper\evaluate_data\evaluate_noise_classification'
    # for i in os.listdir(path):
    #     base = fr'{path}\{i}\classification_noise_test'
    #     gt = f'{path}\\{i}\\GT.json'
    #     print(gt)
    #     for r in [1, 2, 5, 10, 20]:
    #         file = fr'{base}\{r}%\{r}%-result.json'
    #         # print(file)
    #         ret = noise_phase(file)
    #         print(f'{i}-{r}%, ', ret)


    # path = r'G:\paper\evaluate_data\evaluate_noise_classification'
    # print(f'dataset, ', 'TP_G, TP_S, TP_M, FP_G, FP_S, FP_M, FN_G, FN_S, FN_M')
    # for i in os.listdir(path):
    #     base = fr'{path}\{i}\classification_noise_test'
    #     gt = f'{path}\\{i}\\GT.json'
    #     # print(gt)
    #     for r in [1, 2, 5, 10, 20]:
    #         file = fr'{base}\{r}%\{r}%-result.json'
    #         ret = classify_detail(gt, file)
    #         print(f'{i}-{r}%, ', ret)

    # path = r'G:\paper\evaluate_data\evaluate_noise_classification'
    # for i in os.listdir(path):
    #     base = fr'{path}\{i}\classification_noise_test'
    #     gt = f'{path}\\{i}\\GT.json'
    #     pred = f'{path}\\{i}\\result.json'
    #     ret = evaluate_detect(gt, pred)
    #     print(f'{i}, {ret}')
        # break
        # print(pred)
        # t = total(pred)
        # a = avg(pred)
        # print(t, ',',a)
    # path = r'G:\paper\evaluate_data\evaluate_noise_classification'
    # for i in os.listdir(path):
    #     base = fr'{path}\{i}\classification_noise_test'
    #     gt = f'{path}\\{i}\\GT.json'
        # pred_raw = f'{path}\\{i}\\result.json'
        # pred_raw = f'{path}\\{i}\\{i}.json'
        # pred_smooth = f'{path}\\{i}\\tracking_output\\result_with_track.json'
        # # pred_smooth = f'G:\\paper\\evaluate_data\\incorrect-data-test\\normal-dataset\\{i}\\tracking_output\\result_with_track.json'
        # # ret_raw = smooth_evaluate(gt, pred_raw)
        # # ret_smooth = smooth_evaluate(gt, pred_smooth)
        # ret_raw = evaluate_detect(gt, pred_raw)
        # ret_smooth = evaluate_detect(gt, pred_smooth)
        # print(f'{i}-raw, {ret_raw}')
        # print(f'{i}-smooth, {ret_smooth}')

    # run_loss_detection()
    # evaluate_raw_segment()
    run_smooth()