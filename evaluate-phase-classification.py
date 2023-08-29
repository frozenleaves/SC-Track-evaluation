import math
import shutil

import pandas as pd
from tqdm import tqdm
import os
import csv
import numpy as np

import json


class Parser(object):

    def __init__(self, sctrack, pcnadeep, GT):
        self.sctrack = pd.read_csv(sctrack)
        self.pcnadeep = pd.read_csv(pcnadeep)
        self.GT = pd.read_csv(GT)


    def compare(self, row_GT: pd.Series, row_predcit: pd.Series):
        """
        Compare the records of GT and predict, and if they match, return the corresponding GT value.
        Otherwise, return False based on the error type
        """
        # Fs = FP + FN
        try:
            gt_phase = row_GT['phase']
        except KeyError:
            gt_phase = row_GT['cell_type']
        try:
            predict_phase = row_predcit['phase']
        except KeyError:
            predict_phase = row_predcit['cell_type']
        record = {'G1': False, 'S': False, 'G2': False, 'M': False, 'FP_G1': False, 'FP_S': False, 'FP_G2': False,
                  'FP_M': False,
                  'FN_G1': False, 'FN_S': False, 'FN_G2': False, 'FN_M': False
                  }
        if predict_phase[-1] == '*':
            if predict_phase[:-1] == gt_phase:
                # if predict_phase[:-1] == 'G1' or  predict_phase[:-1] == 'G2':
                record[predict_phase[:-1]] = True
            else:
                record['FP_' + predict_phase[:-1]] = True
                record['FN_' + gt_phase] = True

        else:
            if predict_phase == gt_phase:
                record[predict_phase] = True
            else:
                record['FP_' + predict_phase] = True
                record['FN_' + gt_phase] = True
        return record

    def recall(self, TP, FN):
        """recall score"""
        return TP / (TP + FN)

    def precision(self, TP, FP):
        return TP / (TP + FP)

    def F1(self, recall, precision):
        return 2 * precision * recall / (precision + recall)

    def evaluate(self):
        """
        evaluate the accuracy for tracking result, compare with ground truth table.
        """
        sctrack_NUM = 0
        sctrack_TP_G1 = 0
        sctrack_TP_S = 0
        sctrack_TP_G2 = 0
        sctrack_TP_M = 0
        sctrack_FP_G1 = 0
        sctrack_FP_S = 0
        sctrack_FP_G2 = 0
        sctrack_FP_M = 0
        sctrack_FN_G1 = 0
        sctrack_FN_S = 0
        sctrack_FN_G2 = 0
        sctrack_FN_M = 0

        pcnadeep_NUM = 0
        pcnadeep_TP_G1 = 0
        pcnadeep_TP_S = 0
        pcnadeep_TP_G2 = 0
        pcnadeep_TP_M = 0
        pcnadeep_FP_G1 = 0
        pcnadeep_FP_S = 0
        pcnadeep_FP_G2 = 0
        pcnadeep_FP_M = 0
        pcnadeep_FN_G1 = 0
        pcnadeep_FN_S = 0
        pcnadeep_FN_G2 = 0
        pcnadeep_FN_M = 0

        mp_sctrack = {}
        mp_pcnadeep = {}
        data_pcnadeep = self.pcnadeep.sort_values(by='frame_index')
        data_sctrack = self.sctrack.sort_values(by='frame_index')
        data_GT = self.GT.sort_values(by='frame_index')
        frames = pd.unique(data_GT['frame_index'])
        for frame in tqdm(frames):
            tmp_pcnadeep_dict = {}
            tmp_sctrack_dict = {}
            tmp_df_GT = data_GT[data_GT['frame_index'] == frame]
            tmp_df_sctrack = data_sctrack[data_sctrack['frame_index'] == frame]
            tmp_df_pcnadeep = data_pcnadeep[data_pcnadeep['frame_index'] == frame]
            for _, row in tmp_df_GT.iterrows():
                matched_sctrack_flag = False
                matched_pcnadeep_flag = False
                gt_phase = row['phase']
                gt_position = (row['center_x'], row['center_y'])
                for _, sctrack_row in tmp_df_sctrack.iterrows():
                    sctrack_position = (sctrack_row['center_x'], sctrack_row['center_y'])
                    dist = math.sqrt(
                        (sctrack_position[1] - gt_position[1]) ** 2 + (sctrack_position[0] - gt_position[0]) ** 2)
                    if dist < 10:
                        matched_sctrack_flag = True
                        tmp_sctrack_dict[gt_position] = sctrack_position
                        match_result = self.compare(row, sctrack_row)
                        # sctrack_NUM += 1
                        if match_result['G1']:
                            sctrack_TP_G1 += 1
                        if match_result['S']:
                            sctrack_TP_S += 1
                        if match_result['G2']:
                            sctrack_TP_G2 += 1
                        if match_result['M']:
                            sctrack_TP_M += 1
                        if match_result['FP_G1']:
                            sctrack_FP_G1 += 1
                        if match_result['FP_S']:
                            sctrack_FP_S += 1
                        if match_result['FP_G2']:
                            sctrack_FP_G2 += 1
                        if match_result['FP_M']:
                            sctrack_FP_M += 1
                        if match_result['FN_G1']:
                            sctrack_FN_G1 += 1
                        if match_result['FN_S']:
                            sctrack_FN_S += 1
                        if match_result['FN_G2']:
                            sctrack_FN_G2 += 1
                        if match_result['FN_M']:
                            sctrack_FN_M += 1
                        break
                if not matched_sctrack_flag:
                    if gt_phase == 'G1':
                        sctrack_FN_G1 += 1
                    if gt_phase == 'S':
                        sctrack_FN_S += 1
                    if gt_phase == 'G2':
                        sctrack_FN_G2 += 1
                    if gt_phase == 'M':
                        sctrack_FN_M += 1
                sctrack_NUM += 1
                for _, pcnadeep_row in tmp_df_pcnadeep.iterrows():
                    pcnadeep_position = (pcnadeep_row['center_x'], pcnadeep_row['center_y'])
                    dist = math.sqrt(
                        (pcnadeep_position[1] - gt_position[1]) ** 2 + (pcnadeep_position[0] - gt_position[0]) ** 2)
                    if dist < 10:
                        matched_pcnadeep_flag = True
                        tmp_pcnadeep_dict[gt_position] = pcnadeep_position
                        # try:
                        #     if pcnadeep_row['phase'][:-1] == '*':
                        #         continue
                        # except KeyError:
                        #     if pcnadeep_row['cell_type'][:-1] == '*':
                        #         continue
                        # pcnadeep_NUM += 1
                        match_result = self.compare(row, pcnadeep_row)
                        if match_result['G1']:
                            pcnadeep_TP_G1 += 1
                        if match_result['S']:
                            pcnadeep_TP_S += 1
                        if match_result['G2']:
                            pcnadeep_TP_G2 += 1
                        if match_result['M']:
                            pcnadeep_TP_M += 1
                        if match_result['FP_G1']:
                            pcnadeep_FP_G1 += 1
                        if match_result['FP_S']:
                            pcnadeep_FP_S += 1
                        if match_result['FP_G2']:
                            pcnadeep_FP_G2 += 1
                        if match_result['FP_M']:
                            pcnadeep_FP_M += 1
                        if match_result['FN_G1']:
                            pcnadeep_FN_G1 += 1
                        if match_result['FN_S']:
                            pcnadeep_FN_S += 1
                        if match_result['FN_G2']:
                            pcnadeep_FN_G2 += 1
                        if match_result['FN_M']:
                            pcnadeep_FN_M += 1
                        break
                if not matched_pcnadeep_flag:
                    if gt_phase == 'G1':
                        pcnadeep_FN_G1 += 1
                    if gt_phase == 'S':
                        pcnadeep_FN_S += 1
                    if gt_phase == 'G2':
                        pcnadeep_FN_G2 += 1
                    if gt_phase == 'M':
                        pcnadeep_FN_M += 1
                pcnadeep_NUM += 1
            mp_sctrack[frame] = tmp_sctrack_dict
            mp_pcnadeep[frame] = tmp_pcnadeep_dict
            # break
        sctrack_result = [sctrack_NUM, sctrack_TP_G1, sctrack_TP_S, sctrack_TP_G2, sctrack_TP_M, sctrack_FP_G1, sctrack_FP_S,
                         sctrack_FP_G2, sctrack_FP_M, sctrack_FN_G1, sctrack_FN_S, sctrack_FN_G2, sctrack_FN_M]
        pcnadeep_result = [pcnadeep_NUM, pcnadeep_TP_G1, pcnadeep_TP_S, pcnadeep_TP_G2, pcnadeep_TP_M, pcnadeep_FP_G1,
                           pcnadeep_FP_S, pcnadeep_FP_G2, pcnadeep_FP_M, pcnadeep_FN_G1, pcnadeep_FN_S, pcnadeep_FN_G2,
                           pcnadeep_FN_M]
        sctrack_recall = [self.recall(sctrack_TP_G1, sctrack_FN_G1), self.recall(sctrack_TP_S, sctrack_FN_S),
                         self.recall(sctrack_TP_G2, sctrack_FN_G2), self.recall(sctrack_TP_M, sctrack_FN_M),
                         self.recall(sum([sctrack_TP_G1, sctrack_TP_S, sctrack_TP_G2, sctrack_TP_M]),
                                     sum([sctrack_FN_G1, sctrack_FN_S, sctrack_FN_G2, sctrack_FN_M]))]
        sctrack_precision = [self.precision(sctrack_TP_G1, sctrack_FP_G1), self.precision(sctrack_TP_S, sctrack_FP_S),
                            self.precision(sctrack_TP_G2, sctrack_FP_G2), self.precision(sctrack_TP_M, sctrack_FP_M),
                            self.precision(sum([sctrack_TP_G1, sctrack_TP_S, sctrack_TP_G2, sctrack_TP_M]),
                                           sum([sctrack_FP_G1, sctrack_FP_S, sctrack_FP_G2, sctrack_FP_M]))]
        sctrack_F1 = [self.F1(*i) for i in zip(sctrack_recall, sctrack_precision)]

        pcnadeep_recall = [self.recall(pcnadeep_TP_G1, pcnadeep_FN_G1), self.recall(pcnadeep_TP_S, pcnadeep_FN_S),
                           self.recall(pcnadeep_TP_G2, pcnadeep_FN_G2), self.recall(pcnadeep_TP_M, pcnadeep_FN_M),
                           self.recall(sum([pcnadeep_TP_G1, pcnadeep_TP_S, pcnadeep_TP_G2, pcnadeep_TP_M]),
                                       sum([pcnadeep_FN_G1, pcnadeep_FN_S, pcnadeep_FN_G2, pcnadeep_FN_M]))]
        pcnadeep_precision = [self.precision(pcnadeep_TP_G1, pcnadeep_FP_G1),
                              self.precision(pcnadeep_TP_S, pcnadeep_FP_S),
                              self.precision(pcnadeep_TP_G2, pcnadeep_FP_G2),
                              self.precision(pcnadeep_TP_M, pcnadeep_FP_M),
                              self.precision(sum([pcnadeep_TP_G1, pcnadeep_TP_S, pcnadeep_TP_G2, pcnadeep_TP_M]),
                                             sum([pcnadeep_FP_G1, pcnadeep_FP_S, pcnadeep_FP_G2, pcnadeep_FP_M]))]
        pcnadeep_F1 = [self.F1(*i) for i in zip(pcnadeep_recall, pcnadeep_precision)]
        sctrack_ret = sctrack_result + sctrack_recall + sctrack_precision + sctrack_F1
        pcnadeep_ret = pcnadeep_result + pcnadeep_recall + pcnadeep_precision + pcnadeep_F1

        # print(f'TP_G1  TP_S  TP_G2  TP_M  Fs\nsctrack: {sctrack_result}\npcnadeep: {pcnadeep_result}', end='')
        # return sctrack_result, pcnadeep_result
        return sctrack_ret, pcnadeep_ret
    

def test():
    gt = r"G:\paper\evaluate_data\incorrect-data-test\normal-dataset\copy_of_1_xy01\5-track-GT.csv"
    sctrack = fr"G:\paper\evaluate_data\incorrect-data-test\normal-dataset\copy_of_1_xy01\tracking_output\track.csv"
    pcnadeep = fr"G:\paper\evaluate_data\incorrect-data-test\normal-dataset\copy_of_1_xy01\track\refined-pcnadeep(CCDeep_format).csv"

    parse = Parser(sctrack, pcnadeep, gt)
    sctrack_ret, pcnadeep_ret = parse.evaluate()
    print('sctrack: ', sctrack_ret)
    print('pcnadeep: ', pcnadeep_ret)