import os
import json
from PIL import Image
import torch.utils.data as data
import torch


class FireSmokeDataset(data.Dataset):
    def __init__(self, path, train=True, img_transform=None, mask_transform=None):
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.path = os.path.join(path, 'train' if train else 'val')

        self.images = list(map(lambda x: os.path.join(self.path, 'images', x),
                               os.listdir(os.path.join(self.path, 'images'))))
        self.masks = list(map(lambda x: os.path.join(self.path, 'masks', x),
                               os.listdir(os.path.join(self.path, 'masks'))))
        self.length = len(self.images)

    def __getitem__(self, index):
        img, msk = self.images[index], self.masks[index]
        image = Image.open(img).convert('RGB')
        with open(msk) as f:
            mask = torch.Tensor(list(json.load(f).values())[0])

        if self.img_transform:
            image = self.img_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

    def __len__(self):
        return self.length
