#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard library
import os
import math
import sys

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(root_dir, "../"))

# 3rd part packages
import tensorflow as tf
from tensorflow.python.framework import tensor_util
import cv2
import numpy as np
import time
# local source

from metrics import accuracy


base_dir = '../data/monuseg/train_val/val/'

pred_sub_dirs = ['result_r0', 'result_r1', 'result_r2', 'result_r3']

gt_dir = os.path.join(base_dir, 'mask')

if __name__ == '__main__':
    all_result_metrics = {}
    metrics_name = ['p_dice', 'p_F1', 'p_iou', 'dice', 'aji', 'iou']

    for thr in [190]:
        print(thr)
        kernel_result_metrics = {}
        for erode_k in [1]:
            print(erode_k)
            result_metrics = {}
            for pred_sub_dir in pred_sub_dirs:
                pred_dir = os.path.join(base_dir, pred_sub_dir)
                result_metrics[pred_sub_dir] = []
                p_dice_sum = 0.0
                p_F1_sum = 0.0
                p_iou_sum = 0.0
                dice_sum = 0.0
                aji_sum = 0.0
                iou_sum = 0.0

                img_num = 0

                for filename in os.listdir(gt_dir):

                    pred_name = os.path.join(pred_dir, filename)
                    gt_name = os.path.join(gt_dir, filename)
                    print(pred_name)
                    sys.stdout.flush()

                    gt = cv2.imread(gt_name, cv2.IMREAD_GRAYSCALE)
                    pred = cv2.imread(pred_name, cv2.IMREAD_GRAYSCALE)
                    # print(gt)
                    # print(pred)

                    _, gt_bin = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
                    _, pred_bin = cv2.threshold(pred, thr, 255, cv2.THRESH_BINARY)

                    kernel = np.ones((erode_k, erode_k), np.uint8)

                    pred_bin = cv2.erode(pred_bin, kernel)

                    # print(gt_bin)

                    single_results = accuracy.compute_metrics(
                        pred_bin, gt_bin, metrics_name)
                    p_dice_sum += single_results['p_dice']
                    p_F1_sum += single_results['p_F1']
                    p_iou_sum += single_results['p_iou']
                    dice_sum += single_results['dice']
                    aji_sum += single_results['aji']
                    iou_sum += single_results['iou']
                    img_num += 1
                result_metrics[pred_sub_dir].append('p_dice_avg : %.4f' % (p_dice_sum / img_num))
                result_metrics[pred_sub_dir].append('p_F1_avg : %.4f' % (p_F1_sum / img_num))
                result_metrics[pred_sub_dir].append('p_iou_avg : %.4f' % (p_iou_sum / img_num))
                result_metrics[pred_sub_dir].append('dice_avg : %.4f' % (dice_sum / img_num))
                result_metrics[pred_sub_dir].append('aji_avg : %.4f' % (aji_sum / img_num))
                result_metrics[pred_sub_dir].append('iou_avg : %.4f' % (iou_sum / img_num))
            kernel_result_metrics[erode_k] = result_metrics
        all_result_metrics[thr] = kernel_result_metrics
    for key, results in all_result_metrics.items():
        print(key)
        for key1, results1 in results.items():
            print(key1)
            for key2, results2 in results1.items():
                for line in results2:
                    print(line)
