#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard library
import os
import sys
sys.path.append("./")
sys.path.insert(0, os.getcwd())
import numpy as np
import cv2
import time

# 3rd part packages
from keras.models import load_model
from argparse import ArgumentParser
from skimage.color import rgb2hed
# local source
from models.linknet import LinkNet
from models.keras_fc_densenet import build_FC_DenseNet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


parser = ArgumentParser()
args = parser.parse_args()


def init_model(patch_size, h5_path, num_classes):
    input_shape = (patch_size, patch_size, 3)
    model = LinkNet(num_classes, input_shape=input_shape)
    model = model.get_model(
        pretrained_encoder=False
    )

    print("--> Loading models: {}".format(h5_path))
    model.load_weights(h5_path)

    # model = build_FC_DenseNet(
    #     model_version='fcdn56', nb_classes=num_classes, final_softmax=False,
    #     input_shape=input_shape, dropout_rate=0.2,
    #     data_format='channels_last')
    #
    # print("--> Loading models: {}".format(h5_path))
    # model.load_weights(h5_path)
    return model


def main(checkpoint_path, val_dir, save_dir, resize_shape):
    # Get command line arguments

    num_class = 1

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_dict = {}

    for i_index, image_name in enumerate(os.listdir(val_dir)):
        print(image_name)
        full_img_name = os.path.join(val_dir, image_name)
        if not os.path.isfile(full_img_name):
            continue

        img_read = np.array(cv2.imread(full_img_name,
                                       cv2.IMREAD_UNCHANGED))

        if resize_shape not in model_dict:
            model_dict[resize_shape] = init_model(resize_shape,
                                                  checkpoint_path,
                                                  num_class)
        model = model_dict[resize_shape]
        print(image_name)
        img_read = cv2.resize(img_read, (resize_shape, resize_shape))

        img = img_read.copy()

        img = img[:, :, 0:3]
        img = img[:, :, ::-1]
        img = np.reshape(img, (1,) + img.shape) / 255.0
        start = time.clock()
        predicts = model.predict(img)
        elapsed = (time.clock() - start)
        print(elapsed)

        print(predicts[0].shape)

        pred_mask = predicts[0][0, :, :, 0]
        pred_mask[pred_mask > 255] = 255
        pred_mask[pred_mask < 0] = 0

        # pred_mask[pred_mask < 0] = 0
        # pred_mask[pred_mask > 255] = 255
        # _, pred_mask_bin = cv2.threshold(pred_mask, args.th, 255,
        #                                  cv2.THRESH_BINARY)
        save_name = os.path.join(save_dir, image_name)
        cv2.imwrite(save_name, pred_mask)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    fold = 'train_val'
    model_name = 'LinkNet.nuclei.train_val.512_loss_0.01_0.01_0.01_0.01_1.0' \
                 '_train_val_r3_resume_point_edge_fake_sobel.last.h5'

    checkpoint_path = os.path.join(
        f'checkpoints/monuseg_ln/{fold}_model',
        model_name
    )
    print(checkpoint_path)

    val_dir = f'data/monuseg/{fold}/val/img/'
    save_dir = f'data/monuseg/{fold}/val/result_r3/'

    print(val_dir)
    print(save_dir)
    resize_shape = 1000

    main(checkpoint_path, val_dir, save_dir, resize_shape)
