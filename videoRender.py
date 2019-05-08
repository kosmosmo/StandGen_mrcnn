import cv2,os,logging
import numpy as np
from standGen import imageProcess,frameObject
import sys
from samples import coco
from mrcnn import utils
from mrcnn import model as modellib
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

input_video = 'video_1.mp4'
capture = cv2.VideoCapture(input_video)
##will replace to frameobject
#fg = cv2.imread('demo/sp.jpeg',-1)
#fgmask = cv2.imread('demo/spMask.jpeg',-1)


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
sp = frameObject.frame('stands/sp/',0.001,1)


fps = 24.0
width = int(capture.get(3))
height = int(capture.get(4))
fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
out = cv2.VideoWriter("new_video.avi", fcc, fps, (width, height))
frameNum = 0

while True:
    ret, frame = capture.read()
    #use mrcnn to generate a mask and fethering the shit out of it
    results = model.detect([frame], verbose=0)
    r = results[0]
    mask = np.uint8(r['masks'][:,:,:1]*255)
    mask = imageProcess.feather(mask,40)[:,:,:1]

    #tf_pose to get neck joint and nose joint
    humans = e.inference(frame, resize_to_default=(w > 0 and h > 0), upsample_size=4)
    p0 = humans[0].body_parts[0]
    p1 = humans[0].body_parts[1]

    #dealing with some number and start to process
    ratio = imageProcess.getRatio(frame,p0,p1,60) #pix value should get from the standObject
    if sp.getRatio() == None:
        sp.setInitRatio(ratio)
        anchor = imageProcess.getTran(sp.getRatio(), frame, p0, p1, sp.getAnchor())
        sp.setInitTran(anchor)
    else:
        anchor = imageProcess.getTran(sp.getRatio(), frame, p0, p1, sp.getAnchor())
        sp.nextFrame(ratio,anchor)


    fg = cv2.imread(sp.getMaster(),-1)
    fgmask = cv2.imread(sp.getMask(),-1)

    resizeFg = imageProcess.resize(fg,sp.getRatio())
    resizeFgmask = imageProcess.resize(fgmask,sp.getRatio())
    mergeFg2Bg = imageProcess.mergeWithAnchor(resizeFg,frame,sp.getTran()[1],sp.getTran()[0],resizeFgmask[:,:,:1])
    newImg = imageProcess.mergePng(mergeFg2Bg,frame,mask,flag=True)

    #out.write(newImg)
    cv2.imwrite('out/test01.'+str(frameNum).zfill(4)+'.jpeg',newImg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(frameNum)
    print (sp.getRatio(),sp.getTran())
    frameNum+=1

capture.release()
cv2.destroyAllWindows()
