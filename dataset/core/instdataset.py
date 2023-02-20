import os

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class InstCOCODataset(Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.annotation = COCO(annotation)
        self.ids = list(sorted(self.annotation.imgs.keys()))

    def __getitem__(self, index):
        annotation = self.annotation
        img_id = self.ids[index]
        ann_ids = annotation.getAnnIds(imgIds=img_id)
        coco_annotation = annotation.loadAnns(ann_ids)
        path = annotation.loadImgs(img_id)[0]["file_name"].split("\\")[-1]
        img = Image.open(os.path.join(self.root, path))

        num_objs = len(coco_annotation)

        boxes = []
        for i in range(num_objs):
            x_min = coco_annotation[i]["bbox"][0]
            y_min = coco_annotation[i]["bbox"][1]
            x_max = x_min + coco_annotation[i]["bbox"][2]
            y_max = y_min + coco_annotation[i]["bbox"][3]
            boxes.append([x_min, y_min, x_max, y_max])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.ones((num_objs,), dtype=torch.int64)

        img_id = torch.tensor([img_id])

        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]["area"])
        areas = torch.as_tensor(areas, dtype=torch.float32)

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)
