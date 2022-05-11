#%%
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn.neighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pytesseract
from PIL import Image
import cv2
import pandas as pd 
import skimage
from pathlib import Path
from common import *
import tqdm

metrics = []
#%%
for image_id in tqdm.tqdm(range(0,100)):
    print("...")
    im, data = load_sample(image_id)

    gray = np.mean(im,axis=-1).astype(np.uint8)
    t, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    skel = skimage.morphology.skeletonize(thresh//255, method="lee")

    piece = (slice(None,None), slice(None,None))
    hough_input = skel[piece]

    radii = np.arange(110,125)/2
    circles = skimage.transform.hough_circle(hough_input, radii, normalize=False)
    _, i, j = np.unravel_index(np.argsort(-circles.ravel()),circles.shape)
    accum, cx, cy, rad = skimage.transform.hough_circle_peaks(circles, radii, min_xdistance=np.min(radii).astype(int), min_ydistance=np.min(radii).astype(int))

    print("hough done")

    # Max diff.
    lim = accum[1+np.argmax(np.abs(np.diff(accum)))]
    fit = accum > lim

    boxes = []
    for x,y, r, f, a in zip(cx, cy, rad, fit, accum):
        # draw =cv2.circle(draw, (int(x),int(y)), int(r), color=(255,0,0))
    # plt.plot(cx[:50],cy[:50],"rx")
        a = a/accum.max()
        if f:
            boxes.append([x-r,y-r, x+r,y+r])
            # plt.gca().add_patch(plt.Circle((x,y), r, color=(a if f else 0,0,0), fill=False, linewidth=a*3))
    
    gt_boxes = np.stack(data["symbols"]["box"])
    gt_classes = np.stack(data["symbols"]["class"]).astype(int)

    instrumentation_boxes = [box for i, box in enumerate(gt_boxes) if gt_classes[i] in { 26, 27, 28, 29, 31 } ]

    metric = detection_metrics(boxes, instrumentation_boxes)
    print(metric)
    metrics.append(metric)

    # plt.imshow(hough_input)
# %%
