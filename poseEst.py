import cv2
import numpy as np
import logging

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

original_image = 'demo/9.jpg'
image = cv2.imread(original_image)
w, h = model_wh('432x368')
e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h))
humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4)
image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
cv2.imshow('tf-pose-estimation result', image)
k = cv2.waitKey(0)
if k == 27:

    cv2.destroyAllWindows()




