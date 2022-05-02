# %%
from skimage.util import view_as_windows
from craft_text_detector import (
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
)
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
import cv2
import pandas as pd 
import skimage
from pathlib import Path

def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")
def rect_to_slice(rect_pts, margin=0):
    """
    Convert cv-style rect to numpy-style slice
    """
    (x0, y0), (x1, y1) = rect_pts

    return (slice(y0-margin, y1+margin), slice(x0-margin, x1+margin))


dbpath = Path("../DigitizePID_Dataset")

imagepath = dbpath /  "image_2"
imageformat = "jpg"
maskpath  = dbpath /  "mask"
maskformat = "png"

def im2mask(image):
    return maskpath / f"{image.stem}_mask.{maskformat}"
def mask2im(mask):
    return imagepath / f"{mask.stem}.{imageformat}"
def im2info(image):
    dfs = {  }
    for file in  (dbpath / image.stem).glob("*.npy"):
        data = np.load(str(file), allow_pickle=True)
        name = file.stem.split("_")[-1]
        dfs[name] = pd.DataFrame(data)
    return dfs

image = imagepath / f"1.{imageformat}"


# load models
refine_net = load_refinenet_model(cuda=True)
craft_net = load_craftnet_model(cuda=True)

for image in imagepath.glob(f"*.{imageformat}"):

    print(image,end=",")

    im = cv2.imread(str(image))

    roi = (slice(250, 4300), slice(375, 5630))
    im = im[roi]

    tr = np.array([375,250])

    data = im2info(image)

    gray = np.mean(im,axis=-1).astype(np.uint8)
    t, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    window = np.array(im.shape[:2])/5
    wh, ww = window
    sh, sw = (window/2).astype(int)

    t = view_as_windows(thresh, (wh,ww), (sh,sw))

    text = np.stack(data["words"].iloc[:,1]).reshape(-1,2,2) - tr
    def draw_rects(img, tl_br_points, color=(255,0,0), **kwargs):
        for p1, p2 in tl_br_points:
            cv2.rectangle(img, p1, p2, color=color, **kwargs)
    def draw_text_boxes(image, color=(255,0,255), thickness=1):
        draw = image.copy()
        draw_rects(draw, text, color=color, thickness=thickness)
        return draw

    outputs = []
    for window in t.reshape(-1,int(wh),int(ww)):
        prediction_result = get_prediction(
            image=window,
            craft_net=craft_net,
            refine_net=refine_net,
            cuda=False,
            poly=False
        )
        outputs.append(prediction_result)

    offsets = np.zeros((*t.shape[:2],2))
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            offsets[i,j] = (sw*j, sh*i)

    offs_boxes = []
    for offs, output in zip(offsets.reshape(-1,2), outputs):
        if len(output["boxes"])>0:
            boxes = output["boxes"] + offs

            offs_boxes.append(boxes)

    boxes = np.vstack(offs_boxes)
    # print("Boxes:")
    print(len(boxes), end=",")
    boxes_nms = non_max_suppression_fast(boxes[:,0::2].reshape(-1,4), overlapThresh=0.4)
    # print("After NMS:")
    # print(boxes_nms.shape, end=",")

    text_mask = np.zeros_like(thresh)

    gt_text_mask = np.zeros_like(thresh)
    gt_text_mask = draw_text_boxes(gt_text_mask, color=255, thickness=-1)

    draw_rects(text_mask, boxes_nms.reshape(-1,2,2), 255, thickness=-1)

    intersection=text_mask & gt_text_mask

    union=text_mask | gt_text_mask
    # print("iou=")
    print(np.count_nonzero(intersection)/np.count_nonzero(union), end=",")
    print("")

    np.save(str(dbpath/image.stem / "detected_text.npy"), boxes_nms)