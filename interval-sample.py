import json
import os
import shutil

import numpy as np
import tifffile


def sample_frames(json_file, output_dir, interval):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sampled_frames = {}
    index = 0
    for filename, frame_data in data.items():
        if index % interval == 0:
            print(filename)
            new_filename = filename[:-8] + "{:04d}.png".format(index // interval)
            print(new_filename)
            frame_data['filename'] = new_filename
            sampled_frames[new_filename] = frame_data
        index += 1

    output_file = os.path.join(output_dir, str(interval * 5)  + '-' + os.path.basename(json_file))
    with open(output_file, 'w') as f:
        json.dump(sampled_frames, f, indent=4)


def sample_files(input_dir, output_dir, interval):
    os.makedirs(output_dir, exist_ok=True)
    index = 0
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.png'):
            if index % interval == 0:
                new_filename = filename[:-8] + "{:04d}.png".format(index // interval)
                src_path = os.path.join(input_dir, filename)
                dst_path = os.path.join(output_dir, new_filename)
                shutil.copyfile(src_path, dst_path)
            index += 1


def select_frame_for_annotation(gap):
    base_save_dir = rf'G:\paper\evaluate_data\{gap*5}min'
    # dirs = ['copy_of_1_xy01', 'copy_of_1_xy19', 'MCF10A_copy02', 'MCF10A_copy11', 'split-copy19', 'src01', 'src06']
    dirs = ['copy_of_1_xy01','src06',  'copy_of_1_xy19', 'MCF10A_copy02', 'MCF10A_copy11']
    base_file_dir = r'G:\paper\evaluate_data'
    for i in dirs:
        file = os.path.join(base_file_dir, i) + '\\result-GT.json'
        pngs = os.path.join(base_file_dir, i) + '\\png'
        out_dir = os.path.join(base_save_dir, i + f'_{5*gap}min')
        if os.path.exists(file):
            sample_frames(file, out_dir, gap)
            # sample_files(pngs, out_dir + '\\png', gap)


def select_tif_stack(raw_dic, raw_mcy, interval=2):
    """Obtaining tif files with different temporal resolutions through interval sampling"""
    dic = tifffile.imread(raw_dic)
    mcy = tifffile.imread(raw_mcy)
    new_dic = []
    new_mcy = []
    for i in range(len(dic)):
        if i % interval == 0:
            new_dic.append(dic[i])
            new_mcy.append(mcy[i])
    return np.array(new_dic), np.array(new_mcy)


def get_evaluate_image():
    base = r'G:\paper\evaluate_data'
    # dirs = ['copy_of_1_xy01', 'copy_of_1_xy19', 'MCF10A_copy02', 'MCF10A_copy11']
    dirs = ['MCF10A_copy02', 'MCF10A_copy11']

    images = []

    for i in dirs:
        dic = os.path.join(base, i) + '\\dic-sub1.tif'
        mcy = os.path.join(base, i) + '\\mcy-sub1.tif'
        images.append([dic, mcy])
    return images


def run_select_tif(gap=2):
    base = r'G:\paper\evaluate_data'
    base_save_dir = rf'G:\paper\evaluate_data\{gap * 5}min'
    # dirs = ['MCF10A_copy02', 'MCF10A_copy11']
    dirs = ['copy_of_1_xy19']
    for i in dirs:
        # dic = os.path.join(base, i) + '\\dic-sub1.tif'
        # mcy = os.path.join(base, i) + '\\mcy-sub1.tif'
        # out_dir = os.path.join(base_save_dir, i + f'_{5 * gap}min')
        # dic_img, mcy_img = select_tif_stack(dic, mcy, interval=gap)
        # tifffile.imwrite(os.path.join(out_dir, 'dic-sbu1.tif'), dic_img)
        # tifffile.imwrite(os.path.join(out_dir, 'mcy-sbu1.tif'), mcy_img)
        # print(out_dir)
        dic = os.path.join(base, i) + '\\dic.tif'
        mcy = os.path.join(base, i) + '\\mcy.tif'
        out_dir = os.path.join(base_save_dir, i + f'_{5 * gap}min')
        dic_img, mcy_img = select_tif_stack(dic, mcy, interval=gap)
        tifffile.imwrite(os.path.join(out_dir, 'dic.tif'), dic_img)
        tifffile.imwrite(os.path.join(out_dir, 'mcy.tif'), mcy_img)
        print(out_dir)

def get_top_items(dictionary, n):
    top_items = {}
    keys = list(dictionary.keys())[:n]
    for key in keys:
        top_items[key] = dictionary[key]
    return top_items

def generate_example_file():
    raw = r"G:\paper\evaluate_data\5min\src06_5min\5-result-GT.json"
    with open(raw) as f:
        data = json.load(f)

    new = get_top_items(data, 20)
    with open(r"C:\Users\frozen\PycharmProjects\SC-Track\notebook\examples\json_annotation\SEG.json", 'w') as f2:
        json.dump(new, f2)
    # with open(r"C:\Users\frozen\PycharmProjects\SC-Track\notebook\examples\json_annotation\SEG.json") as f:
    #     data = f.readlines()
    #     new_data = data[0].replace('phase', 'cell_type')
    # with open(r"C:\Users\frozen\PycharmProjects\SC-Track\notebook\examples\json_annotation\SEG2.json", 'w') as f:
    #     f.write(new_data)

if __name__ == '__main__':
    # generate_example_file()
    # for i in range(2, 7):
    #     print(i)
    #     select_frame_for_annotation(i)
    # imgs = get_evaluate_image()
    # print(imgs)
    # select_tif_stack(*imgs[0])
    #     run_select_tif(i)
    count = 0
    for i in range(577):
        if i % 4 == 0:
            count += 1
    print(count)