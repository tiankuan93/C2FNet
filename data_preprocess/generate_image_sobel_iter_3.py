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
from skimage import filters
from scipy import misc
from skimage.color import rgb2hed
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
         map_dis=50):
    print('generate_image_sobel_iter_3')
    check_and_creat(edge_supplement_path)
    check_and_creat(edge_sobel_path)
    # check_and_creat(edge_dilate_path)

    for file_name in os.listdir(image_path):
        image_name = os.path.join(image_path, file_name)
        label_name = os.path.join(label_path, file_name)
        pred_name = os.path.join(pred_mask_path, file_name)

        image_np = cv2.imread(image_name, cv2.IMREAD_COLOR)
        label_np = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)

        pred_np = cv2.imread(pred_name, cv2.IMREAD_GRAYSCALE)

        _, pred_np_bin = cv2.threshold(pred_np, 127, 255, cv2.THRESH_BINARY)

        label_regions = get_mask_regions(label_np)

        label_regions_center = get_label_point(image_np.shape, label_regions)

        # fuse Voronoi edge and dilated points
        label_point_dilated = morphology.dilation(
            label_regions_center,
            morphology.disk(3))

        pred_np_bin = get_pred_trans_label(pred_np_bin,
                                           label_point_dilated)

        img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        img_hed = rgb2hed(img_rgb)
        print(img_hed)
        print(img_hed.shape)

        img_hed_h = img_hed[:, :, 0]

        img_hed_norm_h = cv2.normalize(
            img_hed_h, None, 0, 255,
            norm_type=cv2.NORM_MINMAX)

        img_hed_norm_h = cv2.bilateralFilter(
            img_hed_norm_h.astype('uint8'),
            9, 35, 35
            )

        edges = filters.sobel(img_hed_norm_h)
        print(np.unique(edges).astype('uint8'))
        edges = cv2.normalize(
            edges, None, 0, 255,
            norm_type=cv2.NORM_MINMAX)
        _, edges_bin = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

        save_name = os.path.basename(image_name)
        edge_sobel_name = os.path.join(edge_sobel_path, save_name)
        cv2.imwrite(edge_sobel_name, edges)

        print(pred_np_bin.dtype)
        print(edges_bin.dtype)

        kernel = np.ones((5, 5), np.uint8)

        pred_np_bin_dilate = cv2.dilate(pred_np_bin, kernel)
        pred_np_bin_erode = cv2.erode(pred_np_bin, kernel)
        # cv2.imwrite('pred_np_bin_dilate.png', pred_np_bin_dilate)

        label_np_dilate_sub_erode = pred_np_bin_dilate - pred_np_bin_erode
        # label_np_dilate_sub_erode = pred_np_bin_dilate - pred_np_bin

        edges_sup = cv2.bitwise_and(
            label_np_dilate_sub_erode,
            edges_bin.astype('uint8'))

        _, edges_sup_bin = cv2.threshold(
            edges_sup, 128, 1,
            cv2.THRESH_BINARY)

        skeleton_sup = morphology.skeletonize(edges_sup_bin)

        skeleton_sup_remove = morphology.remove_small_objects(
            skeleton_sup, 5,
            connectivity=2)

        #######################
        # pred_np_bin_dilate = cv2.dilate(pred_np_bin, kernel)
        # edges_or = cv2.bitwise_or(pred_np_bin, edges_bin.astype('uint8'))
        # edges_dilate = cv2.bitwise_and(pred_np_bin_dilate, edges_or)
        # edges_dilate = cv2.bitwise_and(edges_bin.astype('uint8'), edges_dilate)
        # edges_sup = edges_dilate - pred_np_bin
        # _, edges_sup_bin = cv2.threshold(
        #     edges_sup, 128, 1,
        #     cv2.THRESH_BINARY)
        # skeleton_sup = morphology.skeletonize(edges_sup_bin)
        # skeleton_sup_remove = morphology.remove_small_objects(
        #     skeleton_sup, 5,
        #     connectivity=2)

        # edge_dilate_name = os.path.join(edge_dilate_path, save_name)
        edge_sup_name = os.path.join(edge_supplement_path, save_name)

        # cv2.imwrite(edge_dilate_name, edges_dilate)
        cv2.imwrite(edge_sup_name, skeleton_sup_remove.astype('uint8')*255)


if __name__ == '__main__':
    base_dir = 'train_r3'

    image_path = os.path.join(base_dir, 'img')
    label_path = os.path.join(base_dir, 'mask')

    # edge_dis_path = os.path.join(base_dir, 'edge_dis')
    # point_dis_path = os.path.join(base_dir, 'point_dis')
    # point_path = os.path.join(base_dir, 'point')
    edge_supplement_path = os.path.join(base_dir, 'edge_supplement')
    edge_sobel_path = os.path.join(base_dir, 'edge_sobel')
    # edge_dilate_path = os.path.join(base_dir, 'edge_dilate')
    pred_mask_path = os.path.join(base_dir, 'result_r2')

    main(image_path, label_path,
         'edge_dis_path',
         'point_dis_path',
         'point_path',
         edge_supplement_path,
         edge_sobel_path=edge_sobel_path,
         pred_mask_path=pred_mask_path)
