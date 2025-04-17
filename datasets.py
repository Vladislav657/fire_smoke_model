import os
import json
from PIL import Image

import torch.utils.data as data

from segment_anything import SamPredictor, sam_model_registry
from utils import get_mask_from_data


class FireSmokeDataset(data.Dataset):
    def __init__(self, path, train=True, transform=None):
        self.transform = transform

        with open(os.path.join(path, "format_train.json" if train else "format_val.json")) as f:
            bboxes = json.load(f)

        self.files = []
        self.targets = []
        self.length = 0

        sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")  # Модель 'vit_b'
        self.predictor = SamPredictor(sam)

        for key, value in bboxes.items():
            self.files.append(os.path.join(path, key))
            self.targets.append(value)
            self.length += 1

    def __getitem__(self, index):
        img, bboxes = str(self.files[index]), self.targets[index]
        mask = get_mask_from_data(img, bboxes, self.predictor)
        img = Image.open(img).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, mask

    def __len__(self):
        return self.length
