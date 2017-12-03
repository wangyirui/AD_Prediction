import numpy as np
import random
import math
from PIL import Image
from skimage.transform import resize
import skimage
import torch


class CustomResize(object):
    def __init__(self, trg_size=(110,110,110)):

        self.trg_size = trg_size

    def __call__(self, img):

        resized_img = self.resize_image(img, self.trg_size)
        return resized_img

    def resize_image(selfs, img, trg_size):
        img_array = np.asarray(img.get_data())
        color_img = skimage.color.gray2rgb(img_array)
        color_img[:, :, :, 0] = img_array
        color_img[:, :, :, 1] = img_array
        color_img[:, :, :, 2] = img_array
        down_sampling = resize(color_img, trg_size, mode='reflect', anti_aliasing=True, preserve_range=True)
        res = down_sampling.astype(np.uint8)

        # type check
        if type(res) != np.ndarray:
            raise "type error!"

        # PIL image cannot handle 3D image, only return ndarray type, which ToTensor accepts
        return res

class CustomToTensor(object):

    def __call__(self, pic):

        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((3, 0, 1, 2)))
            # backward compatibility
            return img.float().div(255)




        