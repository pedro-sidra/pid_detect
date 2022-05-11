# %%
import numpy as np 
import matplotlib.pyplot as plt 
import pytesseract
from PIL import Image
import cv2
import pandas as pd 
import skimage
from pathlib import Path

# %%
from common import *

results = []
for image_id in range(500):
    im, data = load_sample(image_id=image_id)

    boxes_nms = data["text"].to_numpy()
    psm = 7

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    t, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    boxes_filtered = []

    for i, r in enumerate(boxes_nms.reshape(-1,2,2)):
        crop = im[rect_to_slice(r, margin=5)]

        h, w = crop.shape[:2]

        tall = h > 1.3*w
        if tall:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

        try:
            text = pytesseract.image_to_string(crop, config=f"--oem 3 --psm {psm}")
        except Exception as e:
            print("Oopsie from tesseract")
            continue
        
        if len(text)>0:
            alpha_percent = alpha_count(text) / len(text)

            if alpha_percent < 0.4 :
                ...
            else:
                boxes_filtered.append(r.flatten())


    gts = np.stack(data["words"]["box"])

    before = (np.round(detection_metrics(boxes_nms,gts, iou_thresh=0.5), 2))
    after = (np.round(detection_metrics(np.stack(boxes_filtered),gts, iou_thresh=0.5),2))

    print(before,after)
    results.append((before, after))
# %%
output = np.stack(results)




# %%
