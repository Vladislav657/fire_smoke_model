import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs_v2
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm

from datasets import FireSmokeDataset
from models import FireSmokeModel
from losses import SoftDiceLoss

# Определяем устройство (GPU, если доступно)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

transforms = tfs_v2.Compose([
    # tfs_v2.CenterCrop(224),
    tfs_v2.ToImage(),
    tfs_v2.ToDtype(torch.float32, scale=True),
])

d_train = FireSmokeDataset("dataset_fire_smoke", transform=transforms)
train_data = data.DataLoader(d_train, batch_size=1, shuffle=True,
                             pin_memory=True)  # pin_memory ускоряет передачу на GPU

model = FireSmokeModel(3, 3).to(device)  # Переносим модель на устройство сразу
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_1 = nn.CrossEntropyLoss().to(device)
loss_2 = SoftDiceLoss().to(device)

epochs = 10
model.train()

for epoch in range(epochs):
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_tqdm:
        x_train, y_train = x_train.to(device), y_train.to(device)  # Переносим данные на устройство

        predict = model(x_train)
        loss = loss_1(predict, y_train) + loss_2(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{epoch + 1}/{epochs}], loss_mean={loss_mean:.3f}")

    # Сохраняем модель с указанием устройства для совместимости
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'fire_smoke_model.tar')

# from utils import *
# import json
# from segment_anything import SamPredictor, sam_model_registry
#
# with open("dataset_fire_smoke/format_train.json") as f:
#     data = json.load(f)
#
# sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")  # Модель 'vit_b'
# predictor = SamPredictor(sam)
# mask = get_mask_from_data("dataset_fire_smoke/train/2fe8d7da-86fc-47fc-a685-8b1ad1dd75c6.jpg",
#                           data["train/2fe8d7da-86fc-47fc-a685-8b1ad1dd75c6.jpg"], predictor,
#                           tfs_v2.ToImage())
# save_mask(mask, "mask.png")

