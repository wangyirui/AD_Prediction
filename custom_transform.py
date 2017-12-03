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
        res = resize(img_array, trg_size, mode='reflect', anti_aliasing=False, preserve_range=True)

        # type check
        if type(res) != np.ndarray:
            raise "type error!"

        # PIL image cannot handle 3D image, only return ndarray type, which ToTensor accepts
        return res

    def rescale_image(self, img, trg_size):
        img_array = np.asarray(img.get_data())
        res = resize(img_array, trg_size, mode='reflect', anti_aliasing=False, preserve_range=True)

        return res

class CustomToTensor(object):
    def __init__(self, network_type):

        self.network_type = network_type

    def __call__(self, pic):

        if isinstance(pic, np.ndarray):
            if self.network_type == "AlexNet":
                # handle numpy array
                img = torch.from_numpy(pic.transpose((2, 0, 1)))
            else:
                img = torch.from_numpy(pic.transpose((2, 0, 1)))  
            # backward compatibility
            return img.float().div(255)




        