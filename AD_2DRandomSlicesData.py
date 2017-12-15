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


class AD_2DRandomSlicesData(Dataset):
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

        AXimageList = axRandomSlice(image)
        CORimageList = corRandomSlice(image)
        SAGimageList = sagRandomSlice(image)

        for img2DList in (AXimageList, CORimageList, SAGimageList):
            for image2D in img2DList:
                image2D = Image.fromarray(image2D.astype(np.uint8), 'RGB')
                if self.transform:
                    image2D = self.transform(image2D)
                sample = {'image': image2D, 'label': label}
                samples.append(sample)
        random.shuffle(samples)
        return samples


def getRandomSlice(image_array, mean_direc, fraction, section, step = 2):
    mean_array = np.ndarray.mean(image_array, axis = mean_direc)
    first_p = list(mean_array).index(filter(lambda x: x>0, mean_array)[0])
    last_p = list(mean_array).index(filter(lambda x: x>0, mean_array)[-1])
    slice_p = int(round(first_p + (last_p - first_p)*fraction))
    slice_2Dimgs = []
    slice_select_0 = None
    slice_select_1 = None
    slice_select_2 = None

    randomShift = random.randint(-18, 18)
    slice_p = slice_p + randomShift
    i = 0
    for slice_i in range(slice_p-step, slice_p+step+1, step):
        slice_select = eval("image_array"+section)/1500.0*255
        exec("slice_select_"+str(i)+"=slice_select")
        i += 1
    slice_2Dimg = np.stack((slice_select_0, slice_select_1, slice_select_2), axis = 2)
    slice_2Dimgs.append(slice_2Dimg)
    return slice_2Dimgs

def axRandomSlice(image):
    image_array = np.array(image.get_data())
    return getRandomSlice(image_array, NON_AX, AX_F, AX_SCETION)


def corRandomSlice(image):
    image_array = np.array(image.get_data())
    return getRandomSlice(image_array, NON_COR, COR_F, COR_SCETION)


def sagRandomSlice(image):
    image_array = np.array(image.get_data())
    return getRandomSlice(image_array, NON_SAG, SAG_F, SAG_SCETION)



def getRandom3Slices(image_array, mean_direc, fraction, section, step = 2):
    mean_array = np.ndarray.mean(image_array, axis = mean_direc)
    first_p = list(mean_array).index(filter(lambda x: x>0, mean_array)[0])
    last_p = list(mean_array).index(filter(lambda x: x>0, mean_array)[-1])
    slice_p = int(round(first_p + (last_p - first_p)*fraction))
    slice_2Dimgs = []
    randomShift = random.sample(xrange(-18,19), 3)
    
    for j in range(len(randomShift)):
        slice_sp = slice_p + randomShift[j]
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


def axRandom3Slices(image_array):
    return getRandom3Slices(image_array, NON_AX, AX_F, AX_SCETION)


def corRandom3Slices(image_array):
    return getRandom3Slices(image_array, NON_COR, COR_F, COR_SCETION)


def sagRandomeSlices(image_array):
    return getRandom3Slices(image_array, NON_SAG, SAG_F, SAG_SCETION)





