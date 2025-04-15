import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F


class DigitDataset(Dataset):
    def __init__(self, root_dir, json_file, transform=None):
        """
        Args:
            root_dir (string): 圖像目錄的路徑
            json_file (string): COCO格式標註文件的路徑
            transform (callable, optional): 可選的圖像轉換
        """
        self.root_dir = root_dir
        self.transform = transform

        # 加載COCO格式的標註
        with open(json_file, "r") as f:
            self.coco_data = json.load(f)

        # 創建圖像ID到標註的映射
        self.img_to_anns = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # 創建圖像ID到圖像信息的映射
        self.img_to_info = {img["id"]: img for img in self.coco_data["images"]}

        # 獲取所有圖像ID
        self.image_ids = list(self.img_to_info.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.img_to_info[img_id]

        # 加載圖像
        img_path = os.path.join(self.root_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        # 獲取該圖像的所有標註
        annotations = self.img_to_anns.get(img_id, [])

        # 準備目標
        boxes = []
        labels = []

        for ann in annotations:
            # COCO格式的bbox是[x, y, width, height]
            bbox = ann["bbox"]
            # 轉換為[x_min, y_min, x_max, y_max]格式
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[0] + bbox[2]
            y_max = bbox[1] + bbox[3]
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann["category_id"])

        # 轉換為tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}

        if self.transform:
            image = self.transform(image)

        return image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize:
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        # 获取原始尺寸
        orig_size = image.size
        # 计算缩放比例
        min_orig_size = float(min(orig_size))
        max_orig_size = float(max(orig_size))
        if max_orig_size / min_orig_size * self.min_size > self.max_size:
            size = self.max_size
        else:
            size = self.min_size
        # 调整图像大小
        image = F.resize(image, size)
        return image, target


class ColorJitter:
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, target):
        image = F.adjust_brightness(
            image, 1.0 + np.random.uniform(-self.brightness, self.brightness)
        )
        image = F.adjust_contrast(
            image, 1.0 + np.random.uniform(-self.contrast, self.contrast)
        )
        image = F.adjust_saturation(
            image, 1.0 + np.random.uniform(-self.saturation, self.saturation)
        )
        image = F.adjust_hue(image, np.random.uniform(-self.hue, self.hue))
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


def get_transform(train):
    """獲取數據轉換"""
    transforms = []

    # 調整圖像大小
    transforms.append(Resize(min_size=800, max_size=1333))

    # 轉換為tensor
    transforms.append(ToTensor())

    if train:
        # 訓練時的數據增強
        transforms.append(ColorJitter())

    # 標準化
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return Compose(transforms)


def collate_fn(batch):
    """自定義batch收集函數"""
    return tuple(zip(*batch))
