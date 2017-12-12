import nibabel as nib
import os
from torch.utils.data import Dataset
import numpy as np
import torch

class AD_Standard_CNN_Dataset(Dataset):
    """labeled Faces in the Wild dataset."""
    
    def __init__(self, root_dir, data_file, transform=None):
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
        
        image_array = customToTensor(np.array(image.get_data()))
        sample = {'image': image_array, 'label': label}
        
        return sample

def customToTensor(pic):
    if isinstance(pic, np.ndarray):
        img = torch.from_numpy(pic)
        img = torch.unsqueeze(img,0)
        # backward compatibility
        return img.float()


