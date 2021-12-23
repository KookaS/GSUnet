import os
import torch
from torch.utils.data import dataset
from torch.utils.data import DataLoader
# transformations that can be used e.g. for data conversion or augmentation
import torchvision.transforms as T
import numpy as np
from PIL import Image

import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

augmentation_visu = False


class VaihingenDataset(dataset.Dataset):
    '''
        Custom Dataset class that loads images and ground truth segmentation
        masks from a directory.
    '''

    # image statistics, calculated in advance as averages across the full
    # training data set
    IMAGE_MEANS = (
        (121.03431026287558, 82.52572736507886,
         81.92368178210943),     # IR-R-G tiles
        (285.34753853934154),                                           # DSM
        (31.005143030549313)                                            # nDSM
    )
    IMAGE_STDS = (
        (54.21029197978022, 38.434924159900554,
         37.040640374137475),    # IR-R-G tiles
        (6.485453035150256),                                            # DSM
        (36.040236155124326)                                            # nDSM
    )

    # label class names
    LABEL_CLASSES = (
        'Impervious', 'Buildings', 'Low Vegetation', 'Tree', 'Car', 'Clutter'
    )

    def __init__(self, data_root, transform=None):
        '''
            Dataset class constructor. Here we initialize the dataset instance
            and retrieve file names (and other metadata, if present) for all the
            images and labels (ground truth semantic segmentation maps).
        '''
        super().__init__()

        self.data_root = data_root

        # find all images. In our case they are listed in a CSV file called
        # "fileList.csv" under the "data_root"
        with open(os.path.join(self.data_root, 'fileList.csv'), 'r') as f:
            lines = f.readlines()
        self.transform = transform
        # parse CSV lines into data tokens: first column is the label file, the
        # remaining ones are the image files
        self.data = []
        for line in lines[1:]:      # skip header
            self.data.append(line.strip().split(','))

    def __len__(self):
        '''
            This function tells the Data Loader how many images there are in
            this dataset.
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''
            Here's where we load, prepare, and convert the images and
            segmentation mask for the data element at the given "idx".
        '''
        item = self.data[idx]

        # load segmentation mask (first column of CSV file)
        labels = Image.open(os.path.join(self.data_root, 'labels', item[0]))
        labels = np.array(labels, dtype=np.int64)   # convert to NumPy array temporarily

        # load all images (remaining columns of CSV file)
        images = [Image.open(os.path.join(self.data_root, 'images', i)) for i in item[1:]]

        # NOTE: at this point it would make sense to perform data augmentation.
        # However, the default augmentations built-in to PyTorch (resp.
        # Torchvision) (i.) only support RGB images; (ii.) only work on the
        # images themselves. In our case, however, we have multispectral data
        # and need to also transform the segmentation mask.
        # This is not difficult to do, but goes beyond the scope of this exercise.
        # For the sake of brevity, we'll leave it out accordingly.
        # What we will have to do, however, is to normalize the image data.
        for i in range(len(images)):
            img = np.array(images[i], dtype=np.float32)                 # convert to NumPy array (very similar to torch.Tensor below)
            img = (img - self.IMAGE_MEANS[i]) / self.IMAGE_STDS[i]      # normalize
            images[i] = img

        # finally, we need to convert our data into the torch.Tensor format. For
        # the images, we already have a "ToTensor" transform available, but we
        # need to concatenate the images together.
        tensors = [T.ToTensor()(i) for i in images]
        tensors = torch.cat(tensors, dim=0).float()         # concatenate along spectral dimension and make sure it's in 32-bit floating point

        # For the labels, we need to convert the PIL image to a torch.Tensor.
        labels = torch.from_numpy(labels).long()            # labels need to be in 64-bit integer format

        return tensors, labels


# we also create a function for the data loader here (see Section 2.6 in Exercise 6)
def load_dataloader(batch_size, split):
    data_root = 'dataset_512x512_full'
    return DataLoader(
        VaihingenDataset(os.path.join(data_root, split)),
        batch_size=batch_size,
        # we shuffle the image order for the training dataset
        shuffle=(split == 'train'),
        num_workers=2                   # perform data loading with two CPU threads
    )


def visualise():
    # discrete color scheme
    # 'Impervious', 'Buildings', 'Low Vegetation', 'Tree', 'Car', 'Clutter'
    #discrete color scheme
    cMap = ListedColormap(['black', 'grey', 'lawngreen', 'darkgreen', 'orange', 'red'])     #  'Impervious', 'Buildings', 'Low Vegetation', 'Tree', 'Car', 'Clutter'

    data_root = 'dataset_512x512_full'
    dataset_train = VaihingenDataset(os.path.join(data_root, 'train'))
    print(dataset_train)


    # 196 is good
    aug_data, aug_trg = dataset_train.__getitem__(196)

    augmentation_visu = False
    if augmentation_visu:
        for i in range(len(aug_data)):
            data_augmented = aug_data[i] 
            label_augmented = aug_trg[i]

            f, axarr = plt.subplots(nrows=1,ncols=4)
            plt.sca(axarr[0]); 
            plt.imshow(data_augmented[:3,...].permute(1,2,0).numpy()); plt.title('NIR-R-G')
            plt.sca(axarr[1]); 
            plt.imshow(data_augmented[3,...].squeeze().numpy()); plt.title('DSM')
            plt.sca(axarr[2]); 
            plt.imshow(data_augmented[4,...].squeeze().numpy()); plt.title('nDSM')
            plt.sca(axarr[3]); 
            cax = plt.imshow(label_augmented.squeeze().numpy(), cmap=cMap)                # target: segmentation mask
            cbar = f.colorbar(cax, ticks=list(range(len(dataset_train.LABEL_CLASSES))))
            cbar.ax.set_yticklabels(list(dataset_train.LABEL_CLASSES))
            plt.title('Target: segmentation mask')
            plt.show()
    else:
        data_augmented = aug_data
        label_augmented = aug_trg
        f, axarr = plt.subplots(nrows=1,ncols=4)
        plt.sca(axarr[0]); 
        plt.imshow(data_augmented[:3,...].permute(1,2,0).numpy()); plt.title('NIR-R-G')
        plt.sca(axarr[1]); 
        plt.imshow(data_augmented[3,...].squeeze().numpy()); plt.title('DSM')
        plt.sca(axarr[2]); 
        plt.imshow(data_augmented[4,...].squeeze().numpy()); plt.title('nDSM')
        plt.sca(axarr[3]); 
        cax = plt.imshow(label_augmented.squeeze().numpy(), cmap=cMap)                # target: segmentation mask
        cbar = f.colorbar(cax, ticks=list(range(len(dataset_train.LABEL_CLASSES))))
        cbar.ax.set_yticklabels(list(dataset_train.LABEL_CLASSES))
        plt.title('Target: segmentation mask')
        plt.show()
