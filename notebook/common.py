from unittest import result
import numpy as np
from functools import lru_cache
import pytesseract
from skimage.util import view_as_windows
import cv2
from pathlib import Path
import pandas as pd
# Malisiewicz et al.

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

def load_sample(image_id):
    image = imagepath / f"{image_id}.{imageformat}"
    im = cv2.imread(str(image))

    # 375, 250
    # 5630, 4300
    roi = (slice(250, 4300), slice(375, 5630))
    im = im[roi]

    tr = np.array([375,250])

    data = im2info(image)

    # Translate lines
    lines = np.stack(data["lines"][1]).reshape(-1,2,2) - tr
    data["lines"][1] = list(lines.reshape(-1,4))
    data["lines"].columns=["name","box","code","type"]

    # Fix text boxes
    text_boxes = np.stack(data["words"].iloc[:,1]).reshape(-1,2,2) - tr
    # Sort X and Y coords inside each rect
    text_boxes = np.sort(text_boxes.reshape(-1,2,2),axis=1)
    text_boxes = text_boxes.reshape(-1,4)

    h, w, _  = im.shape

    valid_words = (text_boxes[:,0]< w) & (text_boxes[:,0]> 0)  & (text_boxes[:,1]< h) & (text_boxes[:,1]> 0) 
    text_boxes = text_boxes[valid_words]
    data["words"] = data["words"].loc[valid_words]

    data["words"][1] = list(text_boxes)
    data["words"].columns=["name","box","code","type"]

    # Translate symbols
    symbols = np.stack(data["symbols"].iloc[:,1]).reshape(-1,2,2) - tr
    data["symbols"][1]=list(symbols.reshape(-1,4))
    data["symbols"].columns=["name","box","class"]

    return im, data
def draw_rects(img, tl_br_points, color=(255,0,0), **kwargs):

    if tl_br_points.ndim!=3:
        tl_br_points = tl_br_points.reshape(-1,2,2)

    for p1, p2 in tl_br_points:
        cv2.rectangle(img, p1, p2, color=color, **kwargs)

def rect_to_slice(rect_pts, margin=0):
    """
    Convert cv-style rect to numpy-style slice
    """
    (x0, y0), (x1, y1) = rect_pts

    return (slice(y0-margin, y1+margin), slice(x0-margin, x1+margin))
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

def alpha_count(text):
    return sum( [1 if c.isalnum() else 0 for c in text] )


from craft_text_detector import (
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
)

refine_net = load_refinenet_model(cuda=True)
craft_net = load_craftnet_model(cuda=True)
results = {}
def cleanup_text(im, img_id=None):
    """
    Cleanup text from DigitizePID Image. Only works for that dataset for now
    """

    if img_id in results:
        return results[img_id]
    
    # This is a hack, only works with DigitizePID images cropped by this script
    window = np.array(im.shape[:2])/5

    # Window sizes
    wh, ww = window
    # Strides
    sh, sw = (window/2).astype(int)

    # Prep image for cnn
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    t, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Patches to pass to cnn. Whole image doesn't fit
    t = view_as_windows(thresh, (wh,ww), (sh,sw))

    # Offsets to transform local bounding boxes into global
    offsets = np.zeros((*t.shape[:2],2))
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            offsets[i,j] = (sw*j, sh*i)

    # Run inference on patches
    outputs = []
    for window in t.reshape(-1,int(wh),int(ww)):
        prediction_result = get_prediction(
            image=window,
            craft_net=craft_net,
            refine_net=refine_net,
            cuda=True,
            poly=False
        )
        outputs.append(prediction_result)

    # Offset the predictions back into the original (full) image
    offs_boxes = []
    for offs, output in zip(offsets.reshape(-1,2), outputs):
        if len(output["boxes"])>0:
            boxes = output["boxes"] + offs
            offs_boxes.append(boxes)

    # Detected text
    boxes = np.vstack(offs_boxes)
    # Non-maximium suppression since windowed approach leads to overlaps
    boxes_nms = non_max_suppression_fast(boxes[:,0::2].reshape(-1,4), overlapThresh=0.4)

    # Cleaned-up
    text_cleanup = im.copy()

    # page segmentation mode for tesseract. 7 = assume single line of text 
    psm = 7
    for i, r in enumerate(boxes_nms.reshape(-1,2,2)):
        crop = im[rect_to_slice(r, margin=5)]

        # Rotate vertical text
        h, w = crop.shape[:2]
        tall = h > 1.3*w
        if tall:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)


        # Detect text from crop
        try:
            text = pytesseract.image_to_string(crop, config=f"--oem 3 --psm {psm}")
        except Exception as e:
            print("Oopsie from tesseract")
            continue
        
        # If text detected
        if len(text)>0:
            # Check that the text is ok
            # TODO: (this only avoids text such as ----||, should do a regex here)
            alpha_percent = alpha_count(text) / len(text)

            # Only cleanup valid text
            if alpha_percent > 0.4 :
                draw_rects(text_cleanup, r, (255,255,255), thickness=-1)
    if img_id is not None:
        results[img_id]= text_cleanup
    return text_cleanup

def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img


# boxA = (Ax1,Ay1,Ax2,Ay2)
# boxB = (Bx1,By1,Bx2,By2)
def boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    return True

def getIntersectionArea(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # intersection area
    return (xB - xA + 1) * (yB - yA + 1)

def getUnionAreas(boxA, boxB, interArea=None):
    area_A = getArea(boxA)
    area_B = getArea(boxB)
    if interArea is None:
        interArea = getIntersectionArea(boxA, boxB)
    return area_A + area_B - interArea

def getArea(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

def iou(boxA, boxB):
    # if boxes dont intersect
    if not boxesIntersect(boxA, boxB):
        return 0
    interArea = getIntersectionArea(boxA, boxB)
    union = getUnionAreas(boxA, boxB, interArea=interArea)
    # intersection over union
    iou = interArea / union
    if not  iou >= 0:
        return 0
    return iou

def detection_metrics(preds, gts, iou_thresh=0.5):

    ious = np.zeros((len(preds), len(gts)))

    for i, pred_box in enumerate(preds):
        for j, gt_box in enumerate(gts):
            ious[i,j]=iou(pred_box, gt_box)

    TP = np.any(ious>iou_thresh, axis=1)
    FP = ~TP
    FN = ~np.any(ious>iou_thresh, axis=0)

    recall = np.sum(TP) / (np.sum(FN) + np.sum(TP))
    precision = np.sum(TP) / (np.sum(FP) + np.sum(TP))
    return precision, recall