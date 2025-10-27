# scripts/image_process.py
# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image, ImageOps
import cv2

def _autocrop_on_white(img_pil, threshold=240):
    """يقصّ الجسم الرئيسي إذا الخلفية بيضاء تقريباً."""
    img = np.array(img_pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # عتبة: كل ما هو أغمق من الأبيض يعتبر جسم
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img_pil
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    cropped = img[y:y+h, x:x+w]
    return Image.fromarray(cropped)

def _pad_to_square(img_pil, bg=(255, 255, 255)):
    """يضيف حواف ليصير مربع بدون تشويه."""
    w, h = img_pil.size
    side = max(w, h)
    return ImageOps.pad(img_pil, (side, side), color=bg, centering=(0.5, 0.5))

def prepare_image(image_or_path, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=None):
    """
    ترجع صورة PIL مربّعة بخلفية بيضاء بسيطة — بدون إزالة خلفية عميقة.
    هذا يكفي لتغذية TripoSG.
    """
    if isinstance(image_or_path, Image.Image):
        img = image_or_path.convert("RGB")
    else:
        if not os.path.isfile(image_or_path):
            raise FileNotFoundError(f"Image not found: {image_or_path}")
        img = Image.open(image_or_path).convert("RGB")

    # قص تلقائي خفيف على فرض الخلفية فاتحة
    img = _autocrop_on_white(img)
    # اجعلها مربّع بخلفية بيضاء
    img = _pad_to_square(img, bg=(255, 255, 255))
    # حجم ثابت مريح
    img = img.resize((768, 768), Image.LANCZOS)
    return img
