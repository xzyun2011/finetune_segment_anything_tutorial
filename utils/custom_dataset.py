
import os
import cv2
from PIL import Image
import numpy as np

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, VOCdevkit_path, txt_name="train.txt", transform=None):
        self.VOCdevkit_path = VOCdevkit_path
        with open(os.path.join(VOCdevkit_path, f"VOC2007/ImageSets/Segmentation/{txt_name}"), "r") as f:
            file_names = f.readlines()
        self.file_names = [name.strip() for name in file_names]
        self.image_dir = os.path.join(self.VOCdevkit_path, "VOC2007/JPEGImages")
        self.image_files = [f"{self.image_dir}/{name}.jpg" for name in self.file_names]
        self.gt_dir = os.path.join(self.VOCdevkit_path, "VOC2007/SegmentationObject")
        self.gt_files = [f"{self.gt_dir}/{name}.png" for name in self.file_names]


    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image_name = image_path.split("/")[-1]
        gt_path = self.gt_files[idx]

        image = cv2.imread(image_path)
        image = image[..., ::-1] ## RGB to BGR
        image = np.ascontiguousarray(image)
        gt = Image.open(gt_path)
        gt = np.array(gt, dtype='uint8')
        gt = np.ascontiguousarray(gt)


        ## 直接用sam的图像预处理转tensor
        # if self.transform:
        #     image = self.transform(image)

        return image, gt, image_name

    @staticmethod
    def custom_collate(batch):
        """ DataLoader中collate_fn,
         图像和gt都用numpy格式，后面会重新转tensor
        """
        images = []
        seg_labels = []
        images_name = []
        for image, gt, image_name in batch:
            images.append(image)
            seg_labels.append(gt)
            images_name.append(image_name)
        images = np.array(images)
        seg_labels = np.array(seg_labels)
        return images, seg_labels, images_name