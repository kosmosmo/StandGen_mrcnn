import cv2,os,logging
import numpy as np
from standGen import imageProcess
import sys
from samples import coco
from mrcnn import utils
from mrcnn import model as modellib
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

original_image = 'demo/2.jpg'
bg = cv2.imread(original_image,-1)
##will replace to frameobject
fg = cv2.imread('demo/kqq.png',-1)


#tf-pose model and configuration crap
logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
w, h = model_wh('432x368')
e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h))


#mask-rcnn model and configuration crap
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
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

"""
image processing shit starts here
"""

#use mrcnn to generate a mask and fether the shit out of it
results = model.detect([bg], verbose=0)
r = results[0]
mask = np.uint8(r['masks'][:,:,:1]*255)
mask = imageProcess.feather(mask,100)[:,:,:1]


#tf_pose to get neck joint and nose joint
humans = e.inference(bg, resize_to_default=(w > 0 and h > 0), upsample_size=4)
p0 = humans[0].body_parts[0]
p1 = humans[0].body_parts[1]

#dealing with some number and start to process
ratio = imageProcess.getRatio(bg,p0,p1,100) #pix value should get from the standObject
anchor = imageProcess.getTran(ratio,bg,p1,p1,(500,200)) #fgCenter value should get from the standObject
resizeFg = imageProcess.resize(fg,ratio)
mergeFg2Bg = imageProcess.mergeWithAnchor(resizeFg,bg,anchor[1],anchor[0])
newImg = imageProcess.mergePng(mergeFg2Bg,bg,mask,flag=True)
cv2.imshow('composited image', newImg)
cv2.waitKey(0)
