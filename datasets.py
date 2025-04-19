import os
import json
from PIL import Image
import torch.utils.data as data
import torch


class FireSmokeDataset(data.Dataset):
    def __init__(self, path, train=True, img_transform=None, mask_transform=None):
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.path = path

        with open(os.path.join(path, "format_train.json" if train else "format_val.json")) as f:
            bboxes = json.load(f)

        self.files = []
        self.length = 0

        for key  in bboxes.keys():
            self.files.append(str(os.path.join(self.path, key)))
            self.length += 1

    def __getitem__(self, index):
        img = self.files[index]
        image = Image.open(img).convert('RGB')
        with open(f"{img.rsplit('.', 1)[0]}.json") as f:
            mask = torch.Tensor(list(json.load(f).values())[0])

        if self.img_transform:
            image = self.img_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

    def __len__(self):
        return self.length
