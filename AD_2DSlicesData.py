import nibabel as nib
import os
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize 
from PIL import Image


AX_F = 0.32
COR_F = 0.56
SAG_F =  0.56
NON_AX = (1, 2)
NON_COR = (0, 2)
NON_SAG = (0, 1)
AX_SCETION = "[slice_i, :, :]"
COR_SCETION = "[:, slice_i, :]"
SAG_SCETION = "[:, :, slice_i]"


class AD_2DSlicesData(Dataset):
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

        if img_label == 'Normal':
            label = 0
        elif img_label == 'AD':
            label = 1
        elif img_label == 'MCI':
            label = 2

        image = sag3Slice(image)
        image = Image.fromarray(image.astype(np.uint8), 'RGB')
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label}
        
        return sample

def getSlice(image_array, mean_direc, fraction, section):
    mean_array = np.ndarray.mean(image_array, axis = mean_direc)
    first_p = list(mean_array).index(filter(lambda x: x>0, mean_array)[0])
    last_p = list(mean_array).index(filter(lambda x: x>0, mean_array)[-1])
    slice_i = int(round(first_p + (last_p - first_p)*fraction))
    slice_select = eval("image_array"+section)/1500.0*255
    #slice_select = cutMargin2D(slice_select)
    slice_2Dimg = np.stack((slice_select,)*3, axis = 2)
    return slice_2Dimg

def getPackedSlices(image_array, mean_direc, fraction, section):
    mean_array = np.ndarray.mean(image_array, axis = mean_direc)
    first_p = list(mean_array).index(filter(lambda x: x>0, mean_array)[0])
    last_p = list(mean_array).index(filter(lambda x: x>0, mean_array)[-1])
    slice_i = int(round(first_p + (last_p - first_p)*fraction))
    slice_p = slice_i
    # Middle slice - R Channel
    slice_select_R = eval("image_array"+section)/1500.0*255
    zero_slice = np.zeros(slice_select_R.shape)
    slice_select_R = np.stack((slice_select_R, zero_slice ,zero_slice), axis = 2)
    slices_G = np.zeros(slice_select_R.shape)
    slices_B = np.zeros(slice_select_R.shape)
    # Above middle slice - G Channel
    for slice_i in range(slice_p - 20, slice_p, 2):
        slice_select_G = eval("image_array"+section)/1500.0*255
        slice_select_G = np.stack((zero_slice, slice_select_G, zero_slice), axis = 2)
        slices_G += slice_select_G*0.1
    # Below middle slice - B Channel
    for slice_i in range(slice_p + 2, slice_p + 22, 2):
        slice_select_B = eval("image_array"+section)/1500.0*255
        slice_select_B = np.stack((zero_slice, zero_slice, slice_select_B), axis = 2)
        slices_B += slice_select_B*0.1
    slice_2Dimg = slice_select_R + slices_G + slices_B
    return slice_2Dimg



def axKeySlice(image):
    image_array = np.array(image.get_data())
    return getSlice(image_array, NON_AX, AX_F, AX_SCETION)


def corKeySlice(image):
    image_array = np.array(image.get_data())
    return getSlice(image_array, NON_COR, COR_F, COR_SCETION)


def sagKeySlice(image):
    image_array = np.array(image.get_data())
    return getSlice(image_array, NON_SAG, SAG_F, SAG_SCETION)


def ax3Slice(image):
    image_array = np.array(image.get_data())
    return getPackedSlices(image_array, NON_SAG, SAG_F, AX_SCETION)


def cor3Slice(image):
    image_array = np.array(image.get_data())
    return getPackedSlices(image_array, NON_COR, COR_F, COR_SCETION)


def sag3Slice(image):
    image_array = np.array(image.get_data())
    return getPackedSlices(image_array, NON_SAG, SAG_F, SAG_SCETION)


def axcosag(image, size = (110, 110)):
    ax_slice_R = axKeySlice(image)[:,:,0]
    cor_slice_G = corKeySlice(image)[:,:,0]
    sag_slice_B = sagKeySlice(image)[:,:,0]
    ax_slice_R = resize(ax_slice_R.astype(np.uint8), size, mode='reflect', preserve_range=True)
    cor_slice_G = resize(cor_slice_G.astype(np.uint8), size, mode='reflect', preserve_range=True)
    sag_slice_B = resize(sag_slice_B.astype(np.uint8), size, mode='reflect', preserve_range=True)
    slice_2Dimg = np.stack((ax_slice_R, cor_slice_G, sag_slice_B), axis = 2)
    return slice_2Dimg

def cutMargin2D(image_2D):
    row_mean = np.ndarray.mean(image_2D, axis = 1)
    print row_mean
    first_R = list(row_mean).index(filter(lambda x: x>0, row_mean)[0])
    last_R = list(row_mean).index(filter(lambda x: x>0, row_mean)[-1])
    col_mean = np.ndarray.mean(image_2D, axis = 0)
    first_C = list(row_mean).index(filter(lambda x: x>0, row_mean)[0])
    last_C = list(row_mean).index(filter(lambda x: x>0, row_mean)[-1])
    r_len = last_R - first_R +1 
    c_len = last_C - first_C +1
    side_len = max(r_len+10, c_len+10)
    
    first_out_R = (last_R + first_R)/2 - side_len/2
    last_out_R = (last_R + first_R)/2 + side_len/2

    first_out_C = (last_C + first_C)/2 - side_len/2
    last_out_C = (last_C + first_C)/2 + side_len/2
    out_image = image_2D[first_out_R:last_out_R, first_out_C:last_out_C]
    return out_image



def plotColorImage(image):
    plt.imshow(image.astype(np.uint8))
    plt.show()

def plotGrayImage(image):
    plt.imshow(image.astype(np.uint8), cmap = 'gray')
    plt.show()






