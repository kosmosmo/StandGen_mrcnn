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

original_image = 'demo/2.jpg'
image = cv2.imread(original_image,-1)
w, h = model_wh('432x368')
e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h))
humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4)
print (humans[0].body_parts[1])
print (humans[0].body_parts[1].x,humans[0].body_parts[1].y)
print (humans[0].body_parts[0])
image = TfPoseEstimator.draw_humans_new(image, humans, imgcopy=False)
cv2.imshow('tf-pose-estimation result', image)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()

bg =  image
fg = cv2.imread('demo/kqq.png',-1)

import cvshit
a = cvshit.frame(humans[0].body_parts[0],humans[0].body_parts[1],(500,200),bg,fg)
a.main()




