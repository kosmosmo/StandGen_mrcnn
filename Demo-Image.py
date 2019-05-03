import cv2
import numpy as np
import os
import sys
from samples import coco
from mrcnn import utils
from mrcnn import model as modellib


original_image = 'demo/2.jpg'
# Use OpenCV to read and show the original image
image = cv2.imread(original_image)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 1
config = InferenceConfig()
model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)
model.load_weights(COCO_MODEL_PATH, by_name=True)
class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]



def apply_mask(image, mask):
    image[:, :, 0] = np.where(
        mask == 0,
        gray_image[:, :],
        image[:, :, 0]
    )
    image[:, :, 1] = np.where(
        mask == 0,
        gray_image[:, :],
        image[:, :, 1]
    )
    image[:, :, 2] = np.where(
        mask == 0,
        gray_image[:, :],
        image[:, :, 2]
    )
    return image

def feather(image,n):
    def check(i,j):
        res = []
        for k in range(4):
            ni = i + dirX[k]
            nj = j + dirY[k]
            if (ni,nj) not in visited and ni>= 0 and nj >= 0 and ni < len(image) and nj < len(image[0]) and image[ni][nj][0] == 0:
                res.append((ni,nj))
        return res
    subted = 255/n
    color = 255 - subted
    dirX = [0,0,1,-1]
    dirY = [1,-1,0,0]
    queue = []
    visited = set()
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j][0] == 255:
                queue += check(i,j)
    while color > 0:
        q2 = []
        for item in queue:
            image[item[0]][item[1]] = np.uint8(color)
            res = check(item[0],item[1])
            q2 += res
            visited |= set(res)
        color -= subted
        queue = q2
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    return blur

def display_instances(image, boxes, masks, ids, names, scores):
    max_area = 0
    n_instances = boxes.shape[0]
    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        square = (y2 - y1) * (x2 - x1)
        label = names[ids[i]]
        if label == 'person':
            if square > max_area:
                max_area = square
                mask = masks[:, :, i]
            else:
                continue
        else:
            continue
    image = apply_mask(image, mask)
    return image

results = model.detect([image], verbose=0)
r = results[0]
#mm = display_mask(image,r['masks'])
mk = np.uint8(r['masks'][:,:,:1]*255)
mm = feather(mk,10)
cv2.imwrite('temp/2mask.jpg',mm)
cv2.imshow('mask', mm)
"""
frame = display_instances(
    image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
)
cv2.imshow('save_image', frame)
k = cv2.waitKey(0)
if k == 27:                 
    cv2.destroyAllWindows()
elif k == ord('s'):        
    cv2.imwrite('save_image.jpg', image)
    cv2.destroyAllWindows()
"""