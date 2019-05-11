import cv2,os,logging,sys
import numpy as np
from standGen import imageProcess,frameObject,mergeOver
import sys
from samples import coco
from mrcnn import utils
from mrcnn import model as modellib
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

input_video = 'video.mp4'
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
sp = frameObject.frame('stands/sp/',0.003,3)
go = frameObject.frame('stands/go/',0,0)
dogo = frameObject.frame('stands/dogo/',0,0)
explotion = frameObject.frame('stands/explotion/',0.003,3)
ora = frameObject.frame('stands/ora/',0.003,3)


fps = 24.0
width = int(capture.get(3))
height = int(capture.get(4))
fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
out = cv2.VideoWriter("new_video.avi", fcc, fps, (width, height))
frameNum = 0
maxint = sys.maxsize
while True:
    ret, frame = capture.read()
    #use mrcnn to generate a mask and fethering the shit out of it
    results = model.detect([frame], verbose=0)
    r = results[0]
    mask = np.uint8(r['masks'][:,:,:1]*255)
    mask = imageProcess.feather(mask,20)[:,:,:1]

    #tf_pose to get neck joint and nose joint
    humans = e.inference(frame, resize_to_default=(w > 0 and h > 0), upsample_size=4)
    p0 = humans[0].body_parts[0]
    p1 = humans[0].body_parts[1]

    #dealing with some number and start to process
    mergeExp = mergeOver.mergeOverTracking(explotion,frame,p0,p1,60,frameNum,frameRange=(0,maxint))
    mergeOra = mergeOver.addOverTracking(ora,mergeExp,p0,p1,60,frameNum,frameRange=(0,maxint))
    mergeFg2Bg = mergeOver.mergeOverTracking(sp,mergeOra,p0,p1,60,frameNum,frameRange=(0,maxint))
    newImg = imageProcess.mergePng(mergeFg2Bg,frame,mask,flag=True)

    # overlay go object
    mergeGo = mergeOver.mergeOverCenter(go,newImg,frameNum,frameRange=(0,maxint))


    #out.write(newImg)
    cv2.imwrite('out/test01.'+str(frameNum).zfill(4)+'.jpeg',mergeGo)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(frameNum)
    print (sp.getRatio(),sp.getTran())
    frameNum+=1

capture.release()
cv2.destroyAllWindows()
