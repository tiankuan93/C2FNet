#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard library
import os
import sys

# 3rd part packages
import shutil
# from argparse import ArgumentParser
# local source
from data_preprocess import generate_itetarion_label_itr_0
from data_preprocess import generate_itetarion_label_itr_1
from data_preprocess import generate_itetarion_label_itr_2
from data_preprocess import generate_image_sobel_iter_3
import train_edge_point
import test_edge_point
import train_edge_point_full


def check_and_creat(save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)


def copy_train_data(train_src_dir, train_itr_dir):
    cmd = 'cp -r %s %s' % (train_src_dir, train_itr_dir)
    os.system(cmd)
    print(cmd)


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '6'
    in_dataset_fold = 'train_val'
    train_full_mask_flag = True

    itr_sum = 4

    r_last_ckpt_name = ''
    in_dataset_name = 'monuseg'
    resize_shape = 1000
    save_checkpoint_path = './checkpoints/monuseg_ln'
    # in_dataset_name = 'TNBC_NucleiSegmentation'
    # resize_shape = 512
    # save_checkpoint_path = './checkpoints/tnbc_dn_0'

    for itr_num in range(itr_sum):

        if itr_num == 0:
            generate_train_label = generate_itetarion_label_itr_0
        elif itr_num == 1:
            generate_train_label = generate_itetarion_label_itr_1
        elif itr_num == 2:
            generate_train_label = generate_itetarion_label_itr_2
        elif itr_num == 3:
            generate_train_label = generate_image_sobel_iter_3
        else:
            raise Exception('itr not allowed')

        in_dataset_itr = itr_num
        in_model_ext = 'point_edge_fake_sobel' \
            if itr_num >= 3 else 'point_edge_fake'

        fold_base_dir = './data/%s/%s' % (
            in_dataset_name,
            in_dataset_fold)
        print('#############')
        print(fold_base_dir)
        sys.stdout.flush()

        # copy train data to itr 0 data
        train_src_dir = os.path.join(fold_base_dir, 'train')
        train_itr_dir = os.path.join(fold_base_dir, 'train_r%s' % str(itr_num))
        print(train_src_dir, train_itr_dir)
        copy_train_data(train_src_dir, train_itr_dir)

        val_src_dir = os.path.join(fold_base_dir, 'val')

        # generate itr 0 train label data

        image_path = os.path.join(train_itr_dir, 'img')
        label_path = os.path.join(train_itr_dir, 'mask')

        edge_dis_path = os.path.join(train_itr_dir, 'edge_dis')
        point_dis_path = os.path.join(train_itr_dir, 'point_dis')
        point_path = os.path.join(train_itr_dir, 'point')
        edge_supplement_path = os.path.join(train_itr_dir, 'edge_supplement')
        edge_sobel_path = os.path.join(train_itr_dir, 'edge_sobel')
        pred_mask_path = ''

        if itr_num == 0:
            val_label_path = os.path.join(val_src_dir, 'mask')
            val_edge_dis_path = os.path.join(val_src_dir, 'edge_dis')
            val_point_dis_path = os.path.join(val_src_dir, 'point_dis')
            val_point_path = os.path.join(val_src_dir, 'point')
            val_edge_supplement_path = os.path.join(val_src_dir, 'edge_supplement')
            copy_train_data(val_label_path, val_edge_dis_path)
            copy_train_data(val_label_path, val_point_dis_path)
            copy_train_data(val_label_path, val_point_path)
            copy_train_data(val_label_path, val_edge_supplement_path)

        print('1.generate_train_pred')
        sys.stdout.flush()
        if itr_num > 0:
            pred_mask_path = os.path.join(train_itr_dir, 'result_r%s' % str(itr_num - 1))
            checkpoint_path = r_last_ckpt_name
            print(pred_mask_path)

            test_edge_point.main(checkpoint_path, image_path, pred_mask_path, resize_shape)
        print('1.generate_train_pred end')
        sys.stdout.flush()

        print('2.generate_train_label')
        sys.stdout.flush()
        print(
            image_path, label_path,
            edge_dis_path,
            point_dis_path,
            point_path,
            edge_supplement_path,
            edge_sobel_path,
            pred_mask_path)
        generate_train_label.main(
            image_path, label_path,
            edge_dis_path,
            point_dis_path,
            point_path,
            edge_supplement_path,
            edge_sobel_path,
            pred_mask_path,
            )
            # map_dis=50)

        print('2.generate_train_label end')
        sys.stdout.flush()

        if itr_num == 3:
            tmp_train_itr_dir = os.path.join(fold_base_dir, 'train_r%s' % str(itr_num - 1))
            tmp_edge_dis_path = os.path.join(tmp_train_itr_dir, 'edge_dis')
            tmp_point_dis_path = os.path.join(tmp_train_itr_dir, 'point_dis')
            tmp_point_path = os.path.join(tmp_train_itr_dir, 'point')
            copy_train_data(tmp_edge_dis_path, edge_dis_path)
            copy_train_data(tmp_point_dis_path, point_dis_path)
            copy_train_data(tmp_point_path, point_path)

        if itr_num == 4:
            tmp_train_itr_dir = os.path.join(
                fold_base_dir, 'train_r%s' % str(itr_num - 1))
            tmp_edge_supplement_path = os.path.join(
                tmp_train_itr_dir, 'edge_supplement')

            copy_train_data(tmp_edge_supplement_path, edge_supplement_path)

        # train itr 0 model
        print('3.train_edge_point')
        sys.stdout.flush()
        in_resume_checkpoint_path = r_last_ckpt_name
        r_last_ckpt_name = train_edge_point.main(
            in_dataset_name, in_dataset_fold,
            in_dataset_itr, in_model_ext,
            in_resume_checkpoint_path,
            save_checkpoint_path=save_checkpoint_path)
        print('3.train_edge_point end')
        sys.stdout.flush()

        print('4.test_edge_point')
        sys.stdout.flush()
        checkpoint_path = r_last_ckpt_name
        val_img_dir = os.path.join(val_src_dir, 'img')
        save_dir = os.path.join(val_src_dir, 'result_r%s' % str(itr_num))
        print(val_img_dir)
        print(save_dir)

        test_edge_point.main(
            checkpoint_path, val_img_dir, save_dir, resize_shape)
        print('4.test_edge_point end')
        sys.stdout.flush()

        if itr_num == 3 and train_full_mask_flag:
            print('5.train_full_mask')
            sys.stdout.flush()

            in_dataset_itr = itr_num
            in_model_ext = ''
            in_resume_checkpoint_path = ''
            r_last_ckpt_name = train_edge_point_full.main(
                in_dataset_name, in_dataset_fold,
                in_dataset_itr, in_model_ext, in_resume_checkpoint_path,
                save_checkpoint_path=save_checkpoint_path)

            print('5.train_full_mask end')
            sys.stdout.flush()

            print('6.test_full_mask')
            sys.stdout.flush()

            checkpoint_path = r_last_ckpt_name
            val_img_dir = os.path.join(val_src_dir, 'img')
            save_dir = os.path.join(val_src_dir, 'result_full_mask')
            print(val_img_dir)
            print(save_dir)

            test_edge_point.main(
                checkpoint_path, val_img_dir, save_dir, resize_shape)

            print('6.test_full_mask end')
            sys.stdout.flush()
