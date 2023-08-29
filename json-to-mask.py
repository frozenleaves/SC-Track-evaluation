import numpy as np
from tqdm import tqdm
import cv2
import json
import os


def ellipse_points(center, rx, ry, num_points=100, theta=0):
    all_x = []
    all_y = []
    for i in range(num_points):
        t = i * 2 * np.pi / num_points
        x = center[0] + rx * np.cos(t) * np.cos(theta) - ry * np.sin(t) * np.sin(theta)
        y = center[1] + rx * np.cos(t) * np.sin(theta) + ry * np.sin(t) * np.cos(theta)
        all_x.append(x)
        all_y.append(y)
    return all_x, all_y

def json2mask(JsonFilePath, mask):
    """
    :param JsonFilePath:  json annotation file path
    :param mask: mask image save dir path
    :return: if convert successful, return True
    """
    annotation = json.load(open(JsonFilePath, 'r'))
    if not os.path.exists(mask):
        os.makedirs(mask)
    for i in tqdm(annotation):
        filename = i.replace('.png', '.tif')
        # filename = i.replace('.tif', '.png')
        regions = annotation[i].get('regions')
        # image_path = os.path.join(img, filename)
        # image = cv2.imread(image_path, -1)  # image = skimage.io.imread(image_path)
        # print(image.shape)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # height, width = image.shape[:2]
        height, width = 2048, 2048
        # height, width = 1024, 1024
        mask_arr = np.zeros((height, width), dtype=np.uint8)
        for region in regions:
            if region['shape_attributes'].get('name') == 'ellipse':
                rx = region['shape_attributes'].get('rx')
                ry = region['shape_attributes'].get('ry')
                cx = region['shape_attributes'].get('cx')
                cy = region['shape_attributes'].get('cy')
                theta = region['shape_attributes'].get('theta')
                all_x, all_y = ellipse_points((cx, cy), rx, ry, num_points=32, theta=theta)
                polygons = {'all_points_x': all_x, 'all_points_y': all_y}
            else:
                polygons = region.get('shape_attributes')
            points = []
            for j in range(len(polygons['all_points_x'])):
                x = int(polygons['all_points_x'][j])
                y = int(polygons['all_points_y'][j])
                points.append((x, y))
            contours = np.array(points)
            cv2.fillConvexPoly(mask_arr, contours, (255, 255, 255))
        save_path = os.path.join(mask, filename)
        cv2.imwrite(save_path, mask_arr)


def json_to_mask_deepcell(annotation, height, width):
    """generate mask for deepcell tracker"""
    regions = annotation.get('regions')
    filename = annotation.get('filename')
    # image_path = os.path.join(img, filename)
    # image = cv2.imread(image_path, -1)  # image = skimage.io.imread(image_path)
    # print(image.shape)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # height, width = image.shape[:2]
    # height, width = 2048, 2048
    # height, width = 1024, 1024

    id_map = {}

    mask_arr = np.zeros((height, width), dtype=np.uint8)
    for region in enumerate(regions):
        if region[1]['shape_attributes'].get('name') == 'ellipse':
            rx = region[1]['shape_attributes'].get('rx')
            ry = region[1]['shape_attributes'].get('ry')
            cx = region[1]['shape_attributes'].get('cx')
            cy = region[1]['shape_attributes'].get('cy')
            theta = region[1]['shape_attributes'].get('theta')
            all_x, all_y = ellipse_points((cx, cy), rx, ry, num_points=32, theta=theta)
            polygons = {'all_points_x': all_x, 'all_points_y': all_y}
        else:
            polygons = region[1].get('shape_attributes')
        points = []
        for j in range(len(polygons['all_points_x'])):
            x = int(polygons['all_points_x'][j])
            y = int(polygons['all_points_y'][j])
            points.append((x, y))
        contours = np.array(points)
        center_x = np.mean(polygons['all_points_x'])
        center_y = np.mean(polygons['all_points_y'])
        id_map[region[0]] = (center_x, center_y)
        cv2.fillConvexPoly(mask_arr, contours, (region[0]))
        # cv2.fillConvexPoly(mask_arr, contours, (255,255,255))
    return mask_arr, id_map