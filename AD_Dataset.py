import nibabel as nib
from torch.utils.data import Dataset

class AD_Dataset(Dataset):
    """labeled Faces in the Wild dataset."""
    
    def __init__(self, root_dir, data_file, transform=None, data_augmentation=False):
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
        self.data_augmentation = data_augmentation
    
    def __len__(self):
        return sum(1 for line in open(self.data_file))
    
    def __getitem__(self, idx):
        df = open(self.data_file)
        lines = df.readlines()
        lst = lines[idx].split()
        image_path = os.path.join(self.root_dir, lst[0])
        image = nib.load(image_path)

        if self.data_augmentation:
            image = data_augmentation(image)
        
        if label == 'Normal':
            label = 0
        elif label == 'AD':
            label = 2
        elif label == 'MCI':
            label = 1

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label}
        
        return sample