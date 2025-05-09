import numpy as np
import os
import torch
from PIL import Image
from segment_anything import SamPredictor


'''функция получения маски с помощью модели SAM'''
def get_mask_from_data(img: str, bboxes: dict, predictor: SamPredictor) -> torch.Tensor:
    # Загрузка изображения через PIL ---
    image_pil = Image.open(img) # Чтение изображения
    image = np.array(image_pil)  # Конвертация в numpy-массив (H, W, 3) в RGB

    # Создание пустой маски (фон = 0) ---
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    predictor.set_image(image)  # Установка изображения для SAM

    # Обработка каждого объекта ---
    for obj in sorted(bboxes, key=lambda x: x['category']):
        # print(obj)
        bbox = obj['bbox']
        class_id = obj['category']

        # Преобразуем bbox в формат [x1, y1, x2, y2]
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        sam_bbox = np.array([x1, y1, x2, y2])

        # Генерация масок SAM для bbox
        masks, _, _ = predictor.predict(box=sam_bbox)

        # Выбор маски с площадью, наиболее близкой к заданной
        best_mask_idx = np.argmin([abs(np.sum(m) - obj['area']) for m in masks])
        selected_mask = masks[best_mask_idx]

        # Добавляем маску в итоговое изображение с учетом класса
        mask[selected_mask > 0] = class_id

    one_hot_masks = np.zeros((3, mask.shape[0], mask.shape[1]), dtype=np.uint8)

    for cls in range(3):
        one_hot_masks[cls] = (mask == cls).astype(np.uint8)

    return torch.from_numpy(one_hot_masks).float()


# Визуализация маски
def save_mask(one_hot_masks: torch.Tensor, save_path: str) -> None:
    one_hot_masks = one_hot_masks.argmax(dim=0)
    one_hot_masks = one_hot_masks.numpy().astype(np.uint8)

    palette = np.array([
        [0, 0, 0],  # Фон (0)
        [0, 0, 255],  # Класс 1 - дым
        [255, 0, 0],  # Класс 2 - огонь
    ])

    # Преобразуем маску в цветное изображение
    colored_mask = palette[one_hot_masks]

    # Сохранение маски ---
    mask_pil = Image.fromarray(colored_mask.astype(np.uint8))
    mask_pil.save(save_path)  # Сохраняем как PNG
