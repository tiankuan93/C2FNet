#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard library
import os

# 3rd part packages
from skimage import io
from skimage.measure import label, regionprops
from skimage import morphology
import numpy as np
import cv2
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from scipy.ndimage.morphology import distance_transform_edt

# local source
from data_preprocess.utils import voronoi_finite_polygons_2d, poly2mask


def get_mask_regions(label_mask):
    image_label = label(label_mask)
    image_regions = regionprops(image_label)
    return image_regions


def get_label_point(image_shape, label_regions):
    label_regions_center = np.zeros((image_shape[0], image_shape[1]),
                                    dtype=np.uint8)
    for region in label_regions:
        box_center = region.centroid
        cent_y = int(box_center[0])
        cent_x = int(box_center[1])
        label_regions_center[
            cent_y,
            cent_x] = 255
    return label_regions_center


def get_voronoi_edge(label_point):
    h, w = label_point.shape

    points = np.argwhere(label_point > 0)
    vor = Voronoi(points)

    regions, vertices = voronoi_finite_polygons_2d(vor)
    box = Polygon([[0, 0], [0, w], [h, w], [h, 0]])
    region_masks = np.zeros((h, w), dtype=np.int16)
    edges = np.zeros((h, w), dtype=np.bool)
    count = 1
    for region in regions:
        polygon = vertices[region]
        # Clipping polygon
        poly = Polygon(polygon)
        poly = poly.intersection(box)
        # print(poly)
        polygon = np.array([list(p) for p in poly.exterior.coords])

        mask = poly2mask(polygon[:, 0], polygon[:, 1], (h, w))
        edge = mask * (~morphology.erosion(mask, morphology.disk(1)))
        edges += edge
        region_masks[mask] = count
        count += 1

    edges = (edges > 0).astype(np.uint8) * 255

    return edges


def get_label_gaussian(label, kernel_size=5, sigma_x=5):
    dst = cv2.GaussianBlur(
        label,
        (kernel_size, kernel_size),
        sigmaX=sigma_x,
        borderType=cv2.BORDER_DEFAULT)
    return dst


def get_pred_trans_label(pred_np, label_regions_center):
    label_regions = get_mask_regions(pred_np)
    pred_trans_label = label_regions_center.copy()
    for region in label_regions:
        coords_y = region.coords[:, 0]
        coords_x = region.coords[:, 1]
        region_left = False
        for (y, x) in zip(coords_y, coords_x):
            if pred_trans_label[y, x] > 0:
                region_left = True
        if region_left:
            pred_trans_label[
                coords_y, coords_x] = 255
    return pred_trans_label


def check_and_creat(save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)


def main(image_path, label_path,
         edge_dis_path,
         point_dis_path,
         point_path,
         edge_supplement_path,
         edge_sobel_path='',
         pred_mask_path='',
         map_dis=20):
    print('generate_iteration_label_itr_1')
    check_and_creat(edge_dis_path)
    check_and_creat(point_dis_path)
    check_and_creat(point_path)
    check_and_creat(edge_supplement_path)

    for file_name in os.listdir(image_path):
        image_name = os.path.join(image_path, file_name)
        label_name = os.path.join(label_path, file_name)
        pred_name = os.path.join(pred_mask_path, file_name)
        image_np = io.imread(image_name)
        label_np = io.imread(label_name)
        pred_np = io.imread(pred_name)

        pred_np[pred_np > 127] = 255
        pred_np[pred_np <= 127] = 0

        cv2.imwrite('../weakly_seg_net/ttt.png', pred_np)

        print(image_name)
        print(np.unique(label_np))
        print(np.unique(pred_np))

        label_regions = get_mask_regions(label_np)

        label_regions_center = get_label_point(image_np.shape, label_regions)

        # fuse Voronoi edge and dilated points
        label_point_dilated = morphology.dilation(
            label_regions_center,
            morphology.disk(3))

        label_point_dilated = get_pred_trans_label(pred_np,
                                                   label_point_dilated)

        edges = get_voronoi_edge(label_regions_center)

        edges_gaussian = get_label_gaussian(edges, kernel_size=15, sigma_x=5)

        point_gaussian = get_label_gaussian(
            label_point_dilated,
            kernel_size=15,
            sigma_x=5)

        edges_dis = distance_transform_edt(255 - edges)
        point_dis = distance_transform_edt(255 - label_point_dilated)

        edges_dis_norm = edges_dis * map_dis
        point_dis_norm = point_dis * map_dis

        save_name = os.path.basename(image_name)
        edge_dis_name = os.path.join(edge_dis_path, save_name)
        point_dis_name = os.path.join(point_dis_path, save_name)
        point_name = os.path.join(point_path, save_name)
        edge_supplement_name = os.path.join(edge_supplement_path, save_name)

        cv2.imwrite(edge_supplement_name, label_point_dilated)
        cv2.imwrite(point_name, label_point_dilated)
        cv2.imwrite(edge_dis_name, 255 - edges_dis_norm)
        cv2.imwrite(point_dis_name, 255 - point_dis_norm)


if __name__ == '__main__':
    base_dir = 'train_r1'

    image_path = os.path.join(base_dir, 'img')
    label_path = os.path.join(base_dir, 'mask')

    edge_dis_path = os.path.join(base_dir, 'edge_dis')
    point_dis_path = os.path.join(base_dir, 'point_dis')
    point_path = os.path.join(base_dir, 'point')
    edge_supplement_path = os.path.join(base_dir, 'edge_supplement')
    pred_mask_path = os.path.join(base_dir, 'result_r0')

    main(image_path, label_path,
         edge_dis_path,
         point_dis_path,
         point_path,
         edge_supplement_path,
         pred_mask_path=pred_mask_path)
