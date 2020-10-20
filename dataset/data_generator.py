import os
import numpy as np
from keras.utils import Sequence
import cv2
import random


class ImgTransform(object):
    def __init__(self,
                 img,
                 point_dis,
                 edge_dis,
                 edge_supplement,
                 scalerange,
                 brightrange,
                 patchsize):
        self.img = img
        self.point_dis = point_dis
        self.edge_dis = edge_dis
        self.edge_supplement = edge_supplement
        self.shape = img.shape
        self.patchsize = patchsize
        self.scale_range = scalerange
        self.bright_range = brightrange

    def random_brightness(self):
        bright_factor = round(random.uniform(*self.bright_range), 2)
        self.img = brightness_augment(self.img, factor=bright_factor)

    def random_rotate(self):
        angle = np.random.randint(4)
        self.img, self.point_dis, self.edge_dis, self.edge_supplement = rotate(
            self.img,
            self.point_dis,
            self.edge_dis,
            self.edge_supplement,
            angle)
        self.shape = self.img.shape
        return

    def random_scaling(self):
        self.img, self.point_dis, self.edge_dis, self.edge_supplement = scaling(
            self.img,
            self.point_dis,
            self.edge_dis,
            self.edge_supplement,
            self.scale_range)
        self.shape = self.img.shape
        return

    def random_color(self):
        gammas = list([0.3, 0.4, 0.5, 0.67, 0.9, 1, 1.2, 1.5, 2, 2.5, 3.5])
        i_gamma = np.random.randint(len(gammas))
        self.img = color_change(self.img, gammas[i_gamma])
        return

    def random_flip(self):
        self.img, self.point_dis, self.edge_dis, self.edge_supplement = flip(
            self.img,
            self.point_dis,
            self.edge_dis,
            self.edge_supplement,
            np.random.randint(2))
        self.shape = self.img.shape
        return


class ExtractPatchesRandom(object):
    def __init__(self,
                 img,
                 point_dis,
                 edge_dis,
                 edge_supplement,
                 patchsize=(1024, 1024),
                 flag_rotate=True,
                 flag_scale=True,
                 flag_flip=True,
                 scalerange=(0.8, 1.2),
                 brightrange=(0.8, 1.2),
                 flag_bright=False,
                 flag_color=True,
                 flag_random=True):
        self.patchsize = patchsize
        self.flag_rotate = flag_rotate
        self.flag_scale = flag_scale
        self.flag_flip = flag_flip
        self.flag_random = flag_random
        self.flag_bright = flag_bright
        transform = ImgTransform(img,
                                 point_dis,
                                 edge_dis,
                                 edge_supplement,
                                 scalerange,
                                 brightrange,
                                 patchsize)
        if flag_bright is True:
            transform.random_brightness()

        if flag_scale is True:
            transform.random_scaling()

        if flag_rotate is True:
            transform.random_rotate()

        if flag_flip is True:
            transform.random_flip()

        if flag_color is True:
            transform.random_color()

        self.img = np.concatenate(
            (transform.img,
             transform.point_dis,
             transform.edge_dis,
             transform.edge_supplement),
            axis=2)
        self.shape = self.img.shape

    def get_start(self):
        cor = []
        for i_cor in range(2):
            cor.append(
                random.randint(0, self.shape[i_cor]-self.patchsize[i_cor]))
        return cor

    def extractpatch(self):
        if self.img.shape[0] < self.patchsize[0] \
                or self.img.shape[1] < self.patchsize[1]:
            self.img = np.lib.pad(
                self.img,
                ((self.patchsize[0]-self.img.shape[0], 10),
                 (self.patchsize[1]-self.img.shape[1], 10),
                 (0, 0)),
                'constant', constant_values=0)
            self.shape = self.img.shape
        if self.flag_random is True:
            cor = self.get_start()
            patch = self.img[cor[0]:self.patchsize[0]+cor[0],
                             cor[1]:self.patchsize[1]+cor[1], :]
        else:
            patch = self.img[0:self.patchsize[0],
                             0:self.patchsize[1], :]
        return patch


class DataGenerator(Sequence):
    def __init__(self,
                 datapath,
                 mode='train',
                 patchsize=(1024, 1024),
                 batch_size=16,
                 outputchannels=1,
                 expandtimes=10,
                 shuffle=True,
                 flag_rotate=True,
                 flag_scale=True,
                 flag_flip=True,
                 brightrange=(1.0, 1.0),
                 scale_range=(0.9, 1.1),
                 flag_bright=False,
                 flag_color=True,
                 flag_random=True,
                 full_sv=False):
        self.datapath = os.path.join(datapath, mode)
        self.mode = mode
        self.batch_size = batch_size
        self.outputchannels = outputchannels
        self.shuffle = shuffle
        self.patchsize = patchsize
        self.flag_rotate = flag_rotate
        self.flag_scale = flag_scale
        self.flag_flip = flag_flip
        self.bright_range = brightrange
        self.scale_range = scale_range
        self.flag_bright = flag_bright
        self.flag_color = flag_color
        self.flag_random = flag_random
        self.filelist = self.__getlist() * expandtimes
        self.full_sv = full_sv

    def __getlist(self):
        filelist = os.listdir(os.path.join(self.datapath, 'img'))
        if self.shuffle is True:
            random.shuffle(filelist)
        return filelist

    def __len__(self):
        return int(np.floor(len(self.filelist) / self.batch_size))

    def __getitem__(self, item):
        patch_c = np.zeros((self.batch_size,
                            self.patchsize[0],
                            self.patchsize[1],
                            3+self.outputchannels), 'float')
        for i_batch in range(self.batch_size):
            patch_c[i_batch, ...] =\
                self.extract_patch(item*self.batch_size+i_batch)

        point_dis = patch_c[..., 3:4]
        edge_dis = patch_c[..., 4:5]
        edge_supplement = patch_c[..., 5:6]
        point = np.zeros(point_dis.shape)
        point[point_dis == 255] = 255

        img = patch_c[..., 0:3] / 255.0

        return img, [point, point_dis, edge_dis, point_dis, edge_supplement]

    def extract_patch(self, index):
        img_name = os.path.join(
            self.datapath,
            'img', self.filelist[index])
        if self.full_sv:
            point_dis_name = os.path.join(
                self.datapath,
                'mask', self.filelist[index])
        else:
            point_dis_name = os.path.join(
                self.datapath,
                'point_dis', self.filelist[index])

        edge_dis_name = os.path.join(
            self.datapath,
            'edge_dis', self.filelist[index])
        edge_supplement_name = os.path.join(
            self.datapath,
            'edge_supplement', self.filelist[index])
            # 'edge_sobel', self.filelist[index])

        img = np.array(cv2.imread(
            img_name,
            cv2.IMREAD_COLOR))
        point_dis = np.array(cv2.imread(
            point_dis_name,
            cv2.IMREAD_GRAYSCALE))
        edge_dis = np.array(cv2.imread(
            edge_dis_name,
            cv2.IMREAD_GRAYSCALE))
        edge_supplement = np.array(cv2.imread(
            edge_supplement_name,
            cv2.IMREAD_GRAYSCALE))
        scale_range = self.scale_range

        img = img[:, :, 0:3]
        img = img[:, :, ::-1]
        # mask[mask > 0] = 255

        p = ExtractPatchesRandom(
            img,
            point_dis,
            edge_dis,
            edge_supplement,
            patchsize=self.patchsize,
            flag_rotate=self.flag_rotate,
            flag_scale=self.flag_scale,
            flag_flip=self.flag_flip,
            scalerange=scale_range,
            brightrange=self.bright_range,
            flag_bright=self.flag_bright,
            flag_color=self.flag_color,
            flag_random=self.flag_random)
        return p.extractpatch()


def scaling(img, point_dis, edge_dis, edge_supplement, scalerange):
    scale = random.uniform(*scalerange)
    shape = np.array(img.shape)*scale

    img = cv2.resize(
        img.astype('uint8'),
        (int(shape[0]),
         int(shape[1])))
    point_dis = cv2.resize(
        point_dis,
        (int(shape[0]),
         int(shape[1])))
    edge_dis = cv2.resize(
        edge_dis,
        (int(shape[0]),
         int(shape[1])))
    edge_supplement = cv2.resize(
        edge_supplement,
        (int(shape[0]),
         int(shape[1])))
    if len(point_dis.shape) == 2:
        point_dis = point_dis[..., np.newaxis]
    if len(edge_dis.shape) == 2:
        edge_dis = edge_dis[..., np.newaxis]
    if len(edge_supplement.shape) == 2:
        edge_supplement = edge_supplement[..., np.newaxis]

    return img, point_dis, edge_dis, edge_supplement


def rotate(image, point_dis, edge_dis, edge_supplement, angle):
    return np.rot90(image, angle), \
           np.rot90(point_dis, angle), \
           np.rot90(edge_dis, angle), \
           np.rot90(edge_supplement, angle)


def color_change(img, gamma):
    lookuptable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookuptable[0, i] = np.clip(pow(i / 255.0,
                                        gamma) * 255.0, 0, 255)
    return cv2.LUT(img, lookuptable)


def flip(image, point_dis, edge_dis, edge_supplement, axis=0):
    if axis == 0:
        image = image[::-1, ...]
        point_dis = point_dis[::-1, ...]
        edge_dis = edge_dis[::-1, ...]
        edge_supplement = edge_supplement[::-1, ...]
    else:
        if len(image.shape) == 3:
            image = image[:, ::-1, ...]
        else:
            image = image[:, ::-1]
        if len(point_dis.shape) == 3:
            point_dis = point_dis[:, ::-1, ...]
        else:
            point_dis = point_dis[:, ::-1]
        if len(edge_dis.shape) == 3:
            edge_dis = edge_dis[:, ::-1, ...]
        else:
            edge_dis = edge_dis[:, ::-1]
        if len(edge_supplement.shape) == 3:
            edge_supplement = edge_supplement[:, ::-1, ...]
        else:
            edge_supplement = edge_supplement[:, ::-1]
    return image, point_dis, edge_dis, edge_supplement


def brightness_augment(img, factor=0.5):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # convert to hsv
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * factor  # scale channel V uniformly
    hsv = np.clip(hsv, 0, 255)
    rgb = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)
    return rgb


if __name__ == '__main__':
    img_name = '009.png'
    save_name = '009_bri.png'
    img = np.array(cv2.imread(
                    img_name,
                    cv2.IMREAD_UNCHANGED))
    img = img[:, :, 0:3]
    img = img[:, :, ::-1]
    print(img.shape)

    new_img = brightness_augment(img, factor=0.8)
    new_img = new_img[:, :, ::-1]
    cv2.imwrite(save_name, new_img)
