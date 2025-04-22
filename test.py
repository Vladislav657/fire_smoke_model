
'''тестирование модели'''

import torchvision.transforms.v2 as tfs_v2

import json
import torch.utils.data as data
import torch
import torchmetrics

from models import ResNetUNet
from utils import *

from datasets import FireSmokeDataset

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

d_test = FireSmokeDataset("dataset_fire_smoke", train=False, img_transform=img_transforms,
                          mask_transform=mask_transforms)
test_data = data.DataLoader(d_test, batch_size=4)

model.eval()

precision = torchmetrics.Precision(num_classes=3, task='multiclass', average='none')
recall = torchmetrics.Recall(num_classes=3, task='multiclass', average='none')
f1 = torchmetrics.F1Score(num_classes=3, task='multiclass', average='none')

pr, rec, f1_score = torch.zeros(3), torch.zeros(3), torch.zeros(3)
batch_num = 0
for x, y in test_data:
    print('\n\nBatch #', batch_num)

    print('predicting...')
    p = torch.argmax(model(x), dim=1).long()
    y = torch.argmax(y, dim=1).long()

    print('counting metrics...')
    pr += precision(p, y)
    rec += recall(p, y)
    f1_score += f1(p, y)

    print('end #', batch_num)
    batch_num += 1


categories = {1: 'smoke', 2: 'fire'}

pr /= batch_num
rec /= batch_num
f1_score /= batch_num

pr = pr.tolist()
rec = rec.tolist()
f1_score = f1_score.tolist()

print('saving metrics...')
metrics = {}
for i in categories.keys():
    metrics[categories[i]] = {'precision': pr[i], 'recall': rec[i], 'f1_score': f1_score[i]}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)
# x = image * 255
# x = x.permute(1,2,0).numpy()
# x = np.clip(x, 0, 255).astype(np.uint8)
# x = Image.fromarray(x)
# x.save("1_23.jpg")
# print(model)
