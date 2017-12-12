import numpy as np
import random
import math
from PIL import Image
from skimage.transform import resize
import skimage
import torch
import matplotlib.pyplot as plt


class CustomResize(object):
    def __init__(self, trg_size):

        self.trg_size = trg_size


    def __call__(self, img):
        resized_img = self.resize_image(img, self.trg_size)
        return resized_img

    def resize_image(self, img_array, trg_size):
        res = resize(img_array, trg_size, mode='reflect', preserve_range=True, anti_aliasing=False)

        # type check
        if type(res) != np.ndarray:
            raise "type error!"

        # PIL image cannot handle 3D image, only return ndarray type, which ToTensor accepts
        return res

class CustomToTensor(object):
    def __init__(self):
        pass

    def __call__(self, pic):

        if isinstance(pic, np.ndarray):
            
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            
            # backward compatibility
            return img.float()




        
