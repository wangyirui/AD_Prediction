import nibabel as nib
import os
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize 
from PIL import Image
import random
import torch

NON_AX = (0, 1)
NON_COR = (0, 2)
NON_SAG = (1, 2)


class AD_Standard_3DRandomPatch(Dataset):
    """labeled Faces in the Wild dataset."""
    
    def __init__(self, root_dir, data_file):
        """
        Args:
            root_dir (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
        """
        self.root_dir = root_dir
        self.data_file = data_file
    
    def __len__(self):
        with open(self.data_file) as df:
            summation = sum(1 for line in df)
        return summation
    
    def __getitem__(self, idx):
        with open(self.data_file) as df:
            lines = df.readlines()
            lst = lines[idx].split()
            img_name = lst[0]
            image_path = os.path.join(self.root_dir, img_name)
            image = nib.load(image_path)

            image_array = np.array(image.get_data())
            patch_samples = getRandomPatches(image_array)
            patch_dict = {"patch": patch_samples}
        return patch_dict


def customToTensor(pic):
    if isinstance(pic, np.ndarray):
        img = torch.from_numpy(pic)
        img = torch.unsqueeze(img,0)
        # backward compatibility
        return img.float()

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

    first_ax = first_ax + 20
    last_ax = last_ax - 5

    ax_samples = [random.randint(first_ax - 3, last_ax - 3) for r in xrange(10000)]
    cor_samples = [random.randint(first_cor - 3, last_cor - 3) for r in xrange(10000)]
    sag_samples = [random.randint(first_sag - 3, last_sag - 3) for r in xrange(10000)]

    for i in range(1000):
        ax_i = ax_samples[i]
        cor_i = cor_samples[i]
        sag_i = sag_samples[i]
        patch = image_array[ax_i-3:ax_i+4, cor_i-3:cor_i+4, sag_i-3:sag_i+4]
        while (np.ndarray.sum(patch) == 0):
            ax_ni = random.randint(first_ax - 3, last_ax - 4)
            cor_ni = random.randint(first_cor - 3, last_cor - 4)
            sag_ni = random.randint(first_sag - 3, last_sag - 4)
            patch = image_array[ax_ni-3:ax_ni+4, cor_ni-3:cor_ni+4, sag_ni-3:sag_ni+4]
        patch = customToTensor(patch)
        patches.append(patch)
    return patches


# plt.imshow(array[i][3,:,:], cmap = 'gray')
# plt.savefig('./section.png', dpi=100)




