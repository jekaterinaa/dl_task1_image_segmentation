import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(256, 256)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size

        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.img_files.sort()

        self.to_tensor = transforms.ToTensor()  # converts HWC [0,255] -> CHW [0,1]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.img_size, Image.BILINEAR)
        img = self.to_tensor(img)

        # load mask, values 0, 80, 160, 240
        mask = Image.open(mask_path)
        mask = mask.resize(self.img_size, Image.NEAREST)
        mask = np.array(mask, dtype=np.int64)

        # convert pixel values back to class indices: 0->0, 80->1, 160->2, 240->3
        mask = mask // 80
        mask = np.clip(mask, 0, 3)

        mask = torch.from_numpy(mask)

        return img, mask