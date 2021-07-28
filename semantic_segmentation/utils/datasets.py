import os
import numpy as np
import cv2
import pickle
from PIL import Image
from torch.utils.data import Dataset


def line_to_paths_fn_nyudv2(x, input_names):
    return x.decode('utf-8').strip('\n').split('\t')

line_to_paths_fn = {'nyudv2': line_to_paths_fn_nyudv2}


class SegDataset(Dataset):
    """Multi-Modality Segmentation dataset.

    Works with any datasets that contain image
    and any number of 2D-annotations.

    Args:
        data_file (string): Path to the data file with annotations.
        data_dir (string): Directory with all the images.
        line_to_paths_fn (callable): function to convert a line of data_file
            into paths (img_relpath, msk_relpath, ...).
        masks_names (list of strings): keys for each annotation mask
                                        (e.g., 'segm', 'depth').
        transform_trn (callable, optional): Optional transform
            to be applied on a sample during the training stage.
        transform_val (callable, optional): Optional transform
            to be applied on a sample during the validation stage.
        stage (str): initial stage of dataset - either 'train' or 'val'.

    """
    def __init__(self, dataset, data_file, data_dir, bpd_dir, input_names, input_mask_idxs,
                 transform_trn=None, transform_val=None, stage='train', ignore_label=None):
        with open(data_file, 'rb') as f:
            datalist = f.readlines()
        self.datalist =  [i[:6].decode("utf-8") + '.png' for i in datalist] #[line_to_paths_fn[dataset](l, input_names) for l in datalist]
        self.root_dir = data_dir
        self.transform_trn = transform_trn
        self.transform_val = transform_val
        self.stage = stage
        self.input_names = input_names
        self.input_mask_idxs = input_mask_idxs
        self.ignore_label = ignore_label
        self.bpds = pickle.load(open(bpd_dir, 'rb'))

    def set_stage(self, stage):
        """Define which set of transformation to use.

        Args:
            stage (str): either 'train' or 'val'

        """
        self.stage = stage

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        
        def read_image(x, key):
            """Simple image reader

            Args:
                x (str): path to image.

            Returns image as `np.array`.

            """
            if key == 'depth':
                img = cv2.imread(x)
                img = cv2.applyColorMap(cv2.convertScaleAbs(255 - img, alpha=1), cv2.COLORMAP_JET)
                return img
            
            if key == 'rgb':
                img_arr = np.array(Image.open(x))
                if len(img_arr.shape) == 2:  # grayscale
                    img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
                return img_arr

        idxs = self.input_mask_idxs
        img_name = os.path.join(self.root_dir, 'rgb', self.datalist[idx])
        msk_name = os.path.join(self.root_dir, 'masks', self.datalist[idx])
        dpt_name = os.path.join(self.root_dir, 'depth', self.datalist[idx])
        bpd_name = self.datalist[idx][:6]
        
        sample = {}
        sample['rgb'] = read_image(img_name, 'rgb')
        sample['mask'] = np.array(Image.open(msk_name))
        sample['depth'] = read_image(dpt_name, 'depth')
        sample['bpd'] = np.tile(self.bpds[bpd_name], (3, 1, 1)).transpose(1, 2, 0)
        # print(102, sample['bpd'].shape)  
        # print(sample['depth'].shape)    
        # print(104, sample['rgb'].shape)    

        sample['inputs'] = self.input_names

        if self.stage == 'train':
            if self.transform_trn:
                sample = self.transform_trn(sample)
        elif self.stage == 'val':
            if self.transform_val:
                sample = self.transform_val(sample)
        del sample['inputs']
        
        return sample 
    
