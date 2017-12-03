import numpy as np
import random
import math
from PIL import Image
from skimage.transform import resize
import skimage
import torch
import matplotlib.pyplot as plt


class CustomResize(object):
    def __init__(self, network_type, trg_size=(110,110,110)):

        self.trg_size = trg_size
        self.network_type = network_type

    def __call__(self, img):

        if self.network_type == "AlexNet":
            resized_img = self.resize_image(img, self.trg_size)
        else:
            resized_img = self.rescale_image(img, self.trg_size)
        return resized_img

    def resize_image(self, img, trg_size):
        img_array = np.asarray(img.get_data())
        # color_img = skimage.color.gray2rgb(img_array)
        # color_img[:, :, :, 0] = img_array
        # color_img[:, :, :, 1] = img_array
        # color_img[:, :, :, 2] = img_array
        down_sampling = resize(img_array, trg_size, mode='reflect', anti_aliasing=False, preserve_range=True)
        res = down_sampling
        print "pause"

        # type check
        if type(res) != np.ndarray:
            raise "type error!"

        # PIL image cannot handle 3D image, only return ndarray type, which ToTensor accepts
        return res

    def rescale_image(self, img, trg_size):
        img_array = np.asarray(img.get_data())
        down_sampling = resize(img_array, trg_size, mode='reflect', anti_aliasing=False, preserve_range=True)
        res = down_sampling
        return res

class CustomToTensor(object):
    def __init__(self, network_type):

        self.network_type = network_type

    def __call__(self, pic):

        if isinstance(pic, np.ndarray):
            if self.network_type == "AlexNet":
                # handle numpy array
                img = torch.from_numpy(pic.transpose((3, 0, 1, 2)))
                print img.shape
            else:
                img = torch.from_numpy(pic)
                img = torch.unsqueeze(img,0)
                print img.shape
            # backward compatibility
            return img.float().div(255)




        