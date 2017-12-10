import nibabel as nib
import os
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize 
from PIL import Image
import random


NON_AX = (1, 2)
NON_COR = (0, 2)
NON_SAG = (0, 1)


class AD_3DRandomPatch(Dataset):
    """labeled Faces in the Wild dataset."""
    
    def __init__(self, root_dir, data_file, transform=None, slice = slice):
        """
        Args:
            root_dir (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            transform (callable, optional): Optional transform to be applied on a sample.
            data_augmentation (boolean): Optional data augmentation.
        """
        self.root_dir = root_dir
        self.data_file = data_file
        self.transform = transform
    
    def __len__(self):
        return sum(1 for line in open(self.data_file))
    
    def __getitem__(self, idx):
        df = open(self.data_file)
        lines = df.readlines()
        lst = lines[idx].split()
        img_name = lst[0]
        image_path = os.path.join(self.root_dir, img_name)
        image = nib.load(image_path)
        image_array = np.array(image.get_data())
        patch_samples = getRandomPatches(image_array)
        return patch_samples


def getRandomPatches(image_array):
    patches = []
    mean_ax = np.ndarray.mean(image_array, axis = NON_AX)
    mean_cor = np.ndarray.mean(image_array, axis = NON_COR)
    mean_sag = np.ndarray.mean(image_array, axis = NON_SAG)

    first_ax = int(round(list(mean_ax).index(filter(lambda x: x>0, mean_ax)[0])))
    last_ax = int(round(list(mean_ax).index(filter(lambda x: x>0, mean_ax)[-1])))
    first_cor = int(round(list(mean_cor).index(filter(lambda x: x>0, mean_cor)[0])))
    last_cor = int(round(list(mean_cor).index(filter(lambda x: x>0, mean_cor)[-1])))
    first_sag = int(round(list(mean_sag).index(filter(lambda x: x>0, mean_sag)[0])))
    last_sag = int(round(list(mean_sag).index(filter(lambda x: x>0, mean_sag)[-1])))

    first_ax = first_ax + 5
    last_ax = last_ax - 10

    ax_samples = random.choice(xrange(first_ax - 5, last_ax - 5), 1000, replace=True)
    cor_samples = random.choice(xrange(first_cor - 5, last_cor - 5), 1000, replace=True)
    sag_samples = random.choice(xrange(first_sag - 5, last_sag - 5), 1000, replace=True)

    for i in range(1000):
        ax_i = ax_samples[i]
        cor_i = cor_samples[i]
        sag_i = sag_samples[i]
        patch = image_array[ax_i-5:ax_i+5, cor_i-5:cor_i+5, sag_i-5:sag_i+5]
        patches.append(patch)

    return patches



image = nib.load("002_S_0295.nii")
image_array = np.array(image.get_data())
array = getRandomPatches(image_array)



