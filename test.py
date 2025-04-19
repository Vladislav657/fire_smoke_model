import torchvision.transforms.v2 as tfs_v2

from models import FireSmokeModel
from utils import *


img_transforms = tfs_v2.Compose([
    tfs_v2.CenterCrop(384),
    tfs_v2.ToImage(),
    tfs_v2.ToDtype(torch.float32, scale=True),
])

img = "val/1_23.jpg"
image = Image.open(os.path.join("dataset_fire_smoke", img))
image = img_transforms(image)

model = FireSmokeModel(3, 3)

st = torch.load("fire_smoke_model.tar", weights_only=True)

model.load_state_dict(st['model_state_dict'])

model.eval()
with torch.no_grad():
    mask = model(image.unsqueeze(0)).squeeze(0)
save_mask(mask, "mask_1_23.jpg")

x = image * 255
x = x.permute(1,2,0).numpy()
x = np.clip(x, 0, 255).astype(np.uint8)
x = Image.fromarray(x)
x.save("1_23.jpg")
