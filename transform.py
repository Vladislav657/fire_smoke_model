
'''
Скрипт использовался для получения сегментационных масок из bounding boxes
'''





import json

from segment_anything import SamPredictor, sam_model_registry
from utils import *


with open("dataset_fire_smoke/format_val.json") as f:
    bboxes = json.load(f)

# img = "train/0ef48d6f-4c61-4c86-914f-7a6a2b70759a.jpg"

sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")  # Модель 'vit_b'
predictor = SamPredictor(sam)

i = 0
for img in bboxes:
    i += 1
    print(f"{i} - {img}: {bboxes[img]}")

    name = img.split('/')[1]
    path = os.path.join("dataset_fire_smoke/val/images", name)
    mask = get_mask_from_data(path, bboxes[img], predictor).tolist()

    with open(f"dataset_fire_smoke/val/masks/{name.rsplit('.', 1)[0]}.json", "w") as f:
        json.dump({img: mask}, f)
