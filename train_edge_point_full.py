#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard library
import os
import sys
sys.path.append("./")
sys.path.insert(0, os.getcwd())

# 3rd part packages
from keras.models import load_model
from keras.optimizers import RMSprop
from keras import metrics
from keras.callbacks import LearningRateScheduler
from local_callbacks import ModelCheckpoint
from argparse import ArgumentParser
# local source
from args_edge_point import get_arguments
from models.linknet import LinkNet
from models.conv2d_transpose import Conv2DTranspose
# from data import generator_edge_point as data_generator
from dataset import data_generator
from metrics import loss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = ArgumentParser()
args = parser.parse_args()


def main(in_dataset_name, in_dataset_fold,
         in_dataset_itr, in_model_ext,
         in_resume_checkpoint_path, save_checkpoint_path):
    # Get command line arguments

    args.dataset_name = in_dataset_name
    args.dataset_fold = in_dataset_fold
    args.dataset_itr = in_dataset_itr
    model_ext = in_model_ext
    resume_checkpoint_path = in_resume_checkpoint_path

    args.mode = 'train'
    args.resume = False
    args.initial_epoch = 0
    args.pretrained_encoder = True
    args.weights_path = './checkpoints/linknet_encoder_weights.h5'
    args.workers = 32
    args.verbose = 1

    args.learning_rate = 5e-4
    args.lr_decay = 0.1
    args.lr_decay_epochs = 200
    args.epochs = 200
    args.batch_size = 16
    args.outputchannels = 3
    args.scale_range = (0.9, 1.1)

    args.brightrange = (0.7, 1.2)
    args.patch_size = 512
    num_classes = 1
    input_shape = (args.patch_size, args.patch_size, 3)

    args.dataset_dir = os.path.join(
        './data/', args.dataset_name, args.dataset_fold)
    args.checkpoint_dir = os.path.join(
        save_checkpoint_path,
        args.dataset_fold + '_model')

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    args.name = 'LinkNet.nuclei.%s.%s.h5' % (
        os.path.basename(os.path.normpath(args.dataset_dir))[:10],
        str(args.patch_size)
    )

    train_generator = data_generator.DataGenerator(
        datapath=args.dataset_dir,
        mode='train_r0',
        patchsize=(args.patch_size, args.patch_size),
        batch_size=args.batch_size,
        outputchannels=args.outputchannels,
        expandtimes=10,
        shuffle=True,
        flag_rotate=True,
        flag_scale=True,
        flag_flip=True,
        brightrange=args.brightrange,
        scale_range=args.scale_range,
        flag_bright=True,
        flag_color=True,
        full_sv=True
    )
    val_generator = data_generator.DataGenerator(
        datapath=args.dataset_dir,
        mode='val',
        patchsize=(args.patch_size, args.patch_size),
        batch_size=2,
        outputchannels=args.outputchannels,
        expandtimes=1,
        shuffle=False,
        flag_rotate=False,
        flag_scale=True,
        flag_flip=False,
        brightrange=args.brightrange,
        scale_range=args.scale_range,
        flag_bright=False,
        flag_color=False,
        flag_random=False
    )

    loss_weight = {
                'out_1': 1.0,
                'out_2': 0.0,
                'out_3': 0.0,
                'out_4': 0.0,
                'out_5': 0.0
            }
    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        args.name[:-3] +
        '_loss_%s_%s_%s_%s_%s_r0_full_mask.h5' % (
            str(loss_weight['out_1']),
            str(loss_weight['out_2']),
            str(loss_weight['out_3']),
            str(loss_weight['out_4']),
            str(loss_weight['out_5']))
    )
    print("--> Checkpoint path: {}".format(checkpoint_path))

    last_ckpt_path = checkpoint_path[:-3] + '.last.h5'
    best_ckpt_path = checkpoint_path[:-3] + '.{epoch:02d}.h5'

    model = None

    if args.mode.lower() in ('train', 'full'):
        if args.resume:
            print("--> Resuming model: {}".format(resume_checkpoint_path))
            model = LinkNet(num_classes, input_shape=input_shape)
            model = model.get_model(
                pretrained_encoder=False
            )

            model.load_weights(resume_checkpoint_path)
            print('load %s weights success!' % resume_checkpoint_path)

        if model is None:
            model = LinkNet(num_classes, input_shape=input_shape)
            model = model.get_model(
                pretrained_encoder=args.pretrained_encoder,
                weights_path=args.weights_path
            )

        print(model.summary())

        # Optimizer: RMSprop
        optim = RMSprop(args.learning_rate)
        for output in model.outputs:
            print(output.name)

        # Compile the model
        # Loss: Categorical crossentropy loss
        model.compile(
            optimizer=optim,
            loss={
                'out_1': loss.mse_loss,
                'out_2': loss.point_dis_loss,
                'out_3': loss.edge_dis_loss,
                'out_4': loss.fake_loss,
                # 'out_5': loss.edge_supplement_online_loss,
                'out_5': loss.edge_supplement_loss,
            },
            loss_weights=loss_weight,
            metrics=[]
        )

        # Set up learining rate scheduler
        def _lr_decay(epoch):
            return args.lr_decay ** (epoch // args.lr_decay_epochs) *\
                   args.learning_rate

        lr_scheduler = LearningRateScheduler(_lr_decay)

        # Checkpoint callback - save the best model
        checkpoint = ModelCheckpoint(
            best_ckpt_path,
            last_ckpt_path,
            monitor='val_loss',
            save_best=True,
            save_last=True,
            mode='min',
            val_dir=args.dataset_dir + 'val/'
        )

        callbacks = [lr_scheduler, checkpoint]

        # Train the model
        model.fit_generator(
            train_generator,
            epochs=args.epochs,
            max_queue_size=24,
            initial_epoch=args.initial_epoch,
            callbacks=callbacks,
            workers=0,
            verbose=args.verbose,
            use_multiprocessing=False,
            validation_data=val_generator
        )
    return last_ckpt_path


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    in_dataset_name = 'monuseg'
    in_dataset_fold = 'train_val'
    in_dataset_itr = 3
    in_model_ext = 'point_edge_fake_sobel'
    in_resume_checkpoint_path = ''
    main(in_dataset_name, in_dataset_fold,
         in_dataset_itr, in_model_ext, in_resume_checkpoint_path)
