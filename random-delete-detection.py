import json
import math
import os.path
import random

inputfile = r'G:\paper\evaluate_data\5min\copy_of_1_xy01_5min\5-result-test-gap3.json'
outputfile = r'G:\paper\evaluate_data\5min\copy_of_1_xy01_5min\5-result-test-gap_random.json'


def random_del_detection(input, output, ratio):
    with open(input) as f:
        data = json.load(f)
        for i in data:
            frame = data[i]
            regions = frame['regions']
            del_num = math.ceil(len(regions) * ratio)
            for i in range(del_num):
                del_index = random.randint(0, len(regions) - 1)
                del regions[del_index]
    with open(output, 'w') as f2:
        json.dump(data, f2)


def test():
    inputfile = r'G:\paper\evaluate_data\5min\copy_of_1_xy01_5min\5-result-GT.json'
    outputfile = r'G:\paper\evaluate_data\5min\copy_of_1_xy01_5min\5-result-test-andom.json'
    ratio = 0.1
    random_del_detection(inputfile, outputfile, ratio)