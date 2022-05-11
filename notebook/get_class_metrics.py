# %%
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


# %%
trained=False
for image_id in range(100):
    im, data = load_sample(image_id)

    def draw_pipelines(image, data=data):
        draw = image.copy()
        solid_lines = np.stack(data["lines"].query("type=='solid'")["box"])
        dashed_lines = np.stack(data["lines"].query("type=='dashed'")["box"])

        draw = cv2.drawContours(draw, solid_lines.reshape(-1,2,2), -1, (255, 255, 0), thickness=2)
        draw = cv2.drawContours(draw, dashed_lines.reshape(-1,2,2), -1, (0, 255, 255), thickness=2)
        return draw

    def draw_detections(image, rects, classes,color=(255,0,0)):
        draw_rects(image, rects, thickness=8, color=color)
        for r,c in zip(rects,classes):
            cv2.putText(image, str(c), r.flatten()[:2], cv2.FONT_HERSHEY_PLAIN, 6, (0,0,255))

    def draw_symbols(image, data=data, color=None, thickness=2):
        draw = image.copy()
        for i, group in data["symbols"].groupby("class"):
            color_ = color or (np.random.rand(3)*255).astype(np.uint8)
            symbols = np.stack(group["box"])
            draw_rects(draw, symbols, color=[int(c) for c in color_], thickness=thickness)
        return draw

    def draw_gt_symbols(image, data=data,color=None, thickness=2):
        symbol_boxes = np.stack(data["symbols"]["box"])
        symbol_classes = np.stack(data["symbols"]["class"]).astype(int)
        draw_detections(image, symbol_boxes, symbol_classes, color=color)

    def draw_text_boxes(image, data=data, color=(255,0,255), thickness=1):
        draw = image.copy()
        text_boxes = np.stack(data["words"]["box"])
        draw_rects(draw, text_boxes, color=color, thickness=thickness)
        return draw


    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    t, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    X = np.stack(data['words']["box"])

    whs = np.abs(np.stack((X[:,0] - X[:,2], X[:,1]-X[:,3]))).T

    text_heights = np.min(whs,axis=1)
    print("Average text height:")
    np.mean(text_heights[text_heights>0])

    def detect_symbols(image):

        if image.ndim == 3:
            gray = np.mean(image,axis=-1).astype(np.uint8)
        else:
            gray=image

        t, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Foreground is smaller than 50% of image
        if np.count_nonzero(thresh) > thresh.size/2:
            thresh = 255-thresh

        skel = skimage.morphology.skeletonize(thresh//255, method="lee")

        kern = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(41,41))
        closing_kern = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(5,5))

        blackhat = cv2.morphologyEx(skel, cv2.MORPH_BLACKHAT, kern)

        blackhat = cv2.morphologyEx(blackhat, cv2.MORPH_OPEN, closing_kern)
        blackhat = cv2.morphologyEx(blackhat, cv2.MORPH_CLOSE, closing_kern)

        contours, hierarchy = cv2.findContours(blackhat*255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        symbol_boxes = []
        for c in contours:
            if cv2.contourArea(c) > 10:
                x,y,w,h =cv2.boundingRect(c)
                symbol_boxes.append([x,y,x+w,y+h])

        return np.stack(symbol_boxes)

    im_cleanup = cleanup_text(im, img_id=image_id)
    draw = im_cleanup.copy()

    symbol_boxes = detect_symbols(im_cleanup)

    draw_rects(draw, np.stack(symbol_boxes).reshape(-1,2,2), thickness=8)

    import mahotas 

    def get_largest_contour(im):
        contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cmax = sorted(contours, key=cv2.contourArea)[-1]
        return cmax

    def zernike_adaptive_centroid(image, degree=8):
        c = get_largest_contour(image)
        (x,y),r = cv2.minEnclosingCircle(c)
        return  mahotas.features.zernike_moments(image, r, degree=degree)

    def rect_to_slice(rect_pts, margin=0):
        """
        Convert cv-style rect to numpy-style slice
        """
        (x0, y0), (x1, y1) = rect_pts

        return (slice(y0-margin, y1+margin), slice(x0-margin, x1+margin))

    def get_square_thresh(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        t, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        return resizeAndPad(thresh, (64,64), padColor=0)

    def get_zernike_features(image):
        thresh = get_square_thresh(image)
        return zernike_adaptive_centroid(thresh)

    crops = [ im_cleanup[rect_to_slice(s.reshape(2,2), margin=15)] for s in symbol_boxes] 
    features = [ get_zernike_features(crop) for crop in crops]

    import re

    def load_training_set(feature_func):
        training_img_ids = range(400,500) 
        templates_path = Path("../templates")

        file_pattern = re.compile(r"im(\d+)_sym(\d+)")

        train_x = []
        train_y = []
        for template in templates_path.glob("**/*.png"):
            file = template.stem
            id, sym = file_pattern.match(file).groups() 

            if int(id) in training_img_ids:
                template_im = cv2.imread(str(template))

                # Features
                train_x.append(feature_func(template_im))
                # Folder is the class
                train_y.append(int(template.parent.stem))

        return np.stack(train_x), np.stack(train_y)


    def eval_results(image, boxes, feature_func, class_pipeline):

        draw = image.copy()
        draw2 = image.copy()

        # Detect
        # symbol_boxes = detect_symbols(im_cleanup)
        # symbol_boxes = np.stack(data["symbols"]["box"])
        crops = [ image[rect_to_slice(s.reshape(2,2), margin=15)] for s in boxes] 
        features = [ feature_func(crop) for crop in crops]

        probas = class_pipeline.predict_proba(features)
        predictions = 1+np.argmax(probas,axis=1)
        confidences = np.max(probas,axis=1)

        draw_gt_symbols(draw2, thickness=8, color=(0,255,0))

        # Draw results
        draw_detections(draw, np.stack(boxes).reshape(-1,2,2), predictions)

        gt_boxes = np.stack(data["symbols"]["box"])
        gt_classes = np.stack(data["symbols"]["class"]).astype(int)

        metric_data = []
        for clas in np.unique(gt_classes):
            gt = gt_boxes[gt_classes==clas]
            pred = boxes[predictions==clas]
            metrics = detection_metrics(gt,pred)
            metric_data.append({"clas":clas, "precision":metrics[0], "recall":metrics[1]})
        df = pd.DataFrame(metric_data)
        return df


    import sklearn.svm

    print("training....")
    if not trained:
        pipe = Pipeline([("scaler", StandardScaler()), ("classifier", sklearn.svm.SVC(probability=True))])

        train_x, train_y = load_training_set(feature_func=get_zernike_features)
        pipe.fit(train_x, train_y)
        trained=True

    df = eval_results(image=im_cleanup, 
                boxes=detect_symbols(im_cleanup),
                feature_func=get_zernike_features,
                class_pipeline=pipe)

    print(df.mean())
    print(df.sort_values(by="recall", ascending=False))
# %%
