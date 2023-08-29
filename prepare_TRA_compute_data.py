
import os

import cv2
import imagesize
import numpy as np
from tifffile import tifffile
import pandas as pd


class Cell(object):
    def __init__(self, frame, cell_id, parent_id, point_x, point_y):
        self.frame = frame
        self.cell_id = cell_id
        self.parent_id = parent_id
        self.point_x = point_x
        self.point_y = point_y

    @property
    def contours(self):
        points = []
        for j in range(len(self.point_x)):
            x = int(self.point_x[j])
            y = int(self.point_y[j])
            points.append((x, y))
        contours = np.array(points)
        return contours


def parse_table(table):
    df = table
    frames = np.unique(df['frame_index'].to_numpy())
    # for index, row in df.iterrows():
    #     print(row['frame_index'])
    for frame in frames:
        new_df = df[df['frame_index'] == frame]
        cell_list = []
        for _, row in new_df.iterrows():
            cell_id = row['cell_id']
            parent_id = row['parent_id']
            point_x = eval(row['mask_of_x_points'])
            point_y = eval(row['mask_of_y_points'])
            cell = Cell(frame, cell_id, parent_id, point_x, point_y)
            cell_list.append(cell)
        yield frame, cell_list


def generate_TRA_file(table, outfile):
    df = table
    cell_ids = set()
    TRA = []

    for i in df['cell_id']:
        cell_ids.add(i)
    cell_ids = sorted(list(cell_ids))
    for cid in cell_ids:
        new_df = df[df['cell_id'] == cid]
        L = int(cid.replace('_', ''))
        B = new_df.iloc[0]['frame_index']
        E = new_df.iloc[-1]['frame_index']
        if cid == new_df.iloc[0]['parent_id']:
            P = 0
        else:
            P = int(new_df.iloc[0]['parent_id'].replace('_', ''))
        line = f'{L} {B} {E} {P}\n'
        TRA.append(line)
    with open(outfile, 'w') as f:
        f.writelines(TRA)


def table_to_mask(table, width, height, output_dir, flag):
    if not os.path.exists(output_dir):
        if flag in ['gt','GT']:
            os.makedirs(os.path.join(output_dir, 'TRA'))
        elif flag in ['res', 'RES']:
            os.mkdir(output_dir)
    for frame, cells in parse_table(table):
        if flag in ['gt','GT']:
            fname = os.path.join(output_dir, f"TRA\\man_track" + f"{frame}".zfill(3) + ".tif")
        elif flag in ['res', 'RES']:
            fname = os.path.join(output_dir, f"mask" + f"{frame}".zfill(3) + ".tif")
        else:
            raise ValueError(flag)
        mask_arr = np.zeros((height, width), dtype=np.uint16)
        for cell in cells:
            contours = cell.contours
            L = int(cell.cell_id.replace('_', ''))
            cv2.fillConvexPoly(mask_arr, contours, L)
        tifffile.imwrite(fname, mask_arr)


def prepare_evaluate_data(root_dir):
    """Generate TRA evaluation dataset from tracking table"""
    path_gt_table = os.path.join(root_dir, r"track-GT.csv")
    path_res_table = os.path.join(root_dir, r"tracking_output\track.csv")
    img = os.path.join(root_dir, r"01.tif")
    width, height = imagesize.get(img)
    table_res = pd.read_csv(path_res_table)
    table_to_mask(table_res, width, height, os.path.join(root_dir, '01_RES'), 'res')
    generate_TRA_file(table_res, os.path.join(root_dir, '01_RES\\res_track.txt'))

    table_gt = pd.read_csv(path_gt_table)
    table_to_mask(table_gt, width, height, os.path.join(root_dir, '01_GT'), 'gt')
    generate_TRA_file(table_gt, os.path.join(root_dir, '01_GT\\TRA\\man_track.txt'))

if __name__ == '__main__':
    # path = r"G:\paper\test\Data2\tracking_output\track.csv"
    # img = r"G:\paper\test\Data2\01.tif"
    # width, height = imagesize.get(img)
    # table = pd.read_csv(path)
    # table_to_mask(table, width, height, r'G:\paper\test\Data2\01_RES', 'res')
    # generate_TRA_file(table, r'G:\paper\test\Data1\02_RES\res_track.txt')
    prepare_evaluate_data(r"G:\paper\test\Data8")
