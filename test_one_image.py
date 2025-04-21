
'''скрипт визуальной проверки качества сегментации'''

import torch
import torchmetrics
import torchvision.transforms.v2 as tfs_v2

from models import ResNetUNet
from utils import *

img_transforms = tfs_v2.Compose([
    tfs_v2.CenterCrop(512),
    tfs_v2.ToImage(),
    tfs_v2.ToDtype(torch.float32, scale=True),
])

mask_transforms = tfs_v2.Compose([
    tfs_v2.CenterCrop(512),
    tfs_v2.ToDtype(torch.float32),
])

model = ResNetUNet(3)
st = torch.load("fire_smoke_model_res_unet.tar", weights_only=True)

model.load_state_dict(st['model_state_dict'])

image = 'sm_001470.jpg'

img = Image.open(f'dataset_fire_smoke/val/images/{image}').convert('RGB')
img = img_transforms(img)

model.eval()
mask = model(img.unsqueeze(0)).squeeze()

save_mask(mask, f'mask_{image}')

x = img * 255
x = x.permute(1,2,0).numpy()
x = np.clip(x, 0, 255).astype(np.uint8)
x = Image.fromarray(x)
x.save(image)
