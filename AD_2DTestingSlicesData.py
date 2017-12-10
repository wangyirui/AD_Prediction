import nibabel as nib
import os
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize 
from PIL import Image
import random


AX_F = 0.32
COR_F = 0.56
SAG_F =  0.56
NON_AX = (1, 2)
NON_COR = (0, 2)
NON_SAG = (0, 1)
AX_SCETION = "[slice_i, :, :]"
COR_SCETION = "[:, slice_i, :]"
SAG_SCETION = "[:, :, slice_i]"


class AD_2DTestingSlicesData(Dataset):
    """labeled Faces in the Wild dataset."""
    
    def __init__(self, root_dir, data_file, transform=None, slice = slice, size = 3):
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
        self.size = size
    
    def __len__(self):
        return sum(1 for line in open(self.data_file))
    
    def __getitem__(self, idx):
        df = open(self.data_file)
        lines = df.readlines()
        lst = lines[idx].split()
        img_name = lst[0]
        img_label = lst[1]
        image_path = os.path.join(self.root_dir, img_name)
        image = nib.load(image_path)
        samples = []
        if img_label == 'Normal':
            label = 0
        elif img_label == 'AD':
            label = 1
        elif img_label == 'MCI':
            label = 2

        AXimageList = None
        CORimageList = None
        SAGimageList = None

        if self.size == 3:
            AXimageList = axKeySlice(image)
            CORimageList = corKeySlice(image)
            SAGimageList = sagKeySlice(image)
        elif self.size == 9:
            AXimageList = ax3Slices(image)
            CORimageList = cor3Slices(image)
            SAGimageList = sag3Slices(image)

        for img2DList in (AXimageList, CORimageList, SAGimageList):
            for image2D in img2DList
                image2D = Image.fromarray(image2D.astype(np.uint8), 'RGB')
                if self.transform:
                    image2D = self.transform(image2D)
                sample = {'image': image2D, 'label': label}
                samples.append(sample)
        assert len(samples) == self.size
        random.shuffle(samples)
        return samples


def getSlice(image_array, mean_direc, fraction, section, step = 2):
    mean_array = np.ndarray.mean(image_array, axis = mean_direc)
    first_p = list(mean_array).index(filter(lambda x: x>0, mean_array)[0])
    last_p = list(mean_array).index(filter(lambda x: x>0, mean_array)[-1])
    slice_p = int(round(first_p + (last_p - first_p)*fraction))
    slice_2Dimgs = []
    slice_select_0 = None
    slice_select_1 = None
    slice_select_2 = None
    i = 0
    for slice_i in range(slice_p-step, slice_p+step+1, step):
        slice_select = eval("image_array"+section)/1500.0*255
        exec("slice_select_"+str(i)+"=slice_select")
        i += 1
    slice_2Dimg = np.stack((slice_select_0, slice_select_1, slice_select_2), axis = 2)
    slice_2Dimgs.append(slice_2Dimg)
    return slice_2Dimgs

def axKeySlice(image):
    image_array = np.array(image.get_data())
    return getSlice(image_array, NON_AX, AX_F, AX_SCETION)


def corKeySlice(image):
    image_array = np.array(image.get_data())
    return getSlice(image_array, NON_COR, COR_F, COR_SCETION)


def sagKeySlice(image):
    image_array = np.array(image.get_data())
    return getSlice(image_array, NON_SAG, SAG_F, SAG_SCETION)



def get3Slices(image_array, mean_direc, fraction, section, step = 2):
    mean_array = np.ndarray.mean(image_array, axis = mean_direc)
    first_p = list(mean_array).index(filter(lambda x: x>0, mean_array)[0])
    last_p = list(mean_array).index(filter(lambda x: x>0, mean_array)[-1])
    slice_p = int(round(first_p + (last_p - first_p)*fraction))
    slice_2Dimgs = []    
    for shift in (-8, 0, 8):
        slice_sp = slice_p + shift
        i = 0
        slice_select_0 = None
        slice_select_1 = None
        slice_select_2 = None
        for slice_i in range(slice_sp-step, slice_sp+step+1, step):
            slice_select = eval("image_array"+section)/1500.0*255
            exec("slice_select_"+str(i)+"=slice_select")
            i += 1
        slice_2Dimg = np.stack((slice_select_0, slice_select_1, slice_select_2), axis = 2)
        slice_2Dimgs.append(slice_2Dimg)
    return slice_2Dimgs


def ax3Slices(image):
    image_array = np.array(image.get_data())
    return get3Slices(image_array, NON_AX, AX_F, AX_SCETION)


def cor3Slices(image):
    image_array = np.array(image.get_data())
    return get3Slices(image_array, NON_COR, COR_F, COR_SCETION)


def sagSlices(image):
    image_array = np.array(image.get_data())
    return get3Slices(image_array, NON_SAG, SAG_F, SAG_SCETION)





