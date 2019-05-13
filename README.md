# Stand Generation

Mask-RCNN and Openpose implementation for VFX rotoscoping, motion tracking and stands summonation.

[Mask R-CNN](https://github.com/matterport/Mask_RCNN) for instance segmentation.

[Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) for human pose estimation.

## Installation

follow the install instructions here. [Mask R-CNN](https://github.com/matterport/Mask_RCNN) and [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation)

### Dependencies

- python3.6
- tensorflow 1.13.1+
- opencv

### Package Install
```bash
pip install blend_modes
```

## Demo

| Input | Output |
|:---------|:--------------------|
| <img src="https://github.com/kosmosmo/StandGen_mrcnn/blob/master/temp/inSM.gif" width="398" height ="224">| <img src="https://github.com/kosmosmo/StandGen_mrcnn/blob/master/temp/outSM.gif" width="398" height ="224">|
| Mask | PoseTracking |
|<img src="https://github.com/kosmosmo/StandGen_mrcnn/blob/master/temp/mask.jpeg" width="398" height ="224">   | <img src="https://github.com/kosmosmo/StandGen_mrcnn/blob/master/temp/track.jpeg" width="398" height ="224">|

| Compositing |
|:---------|
| ![](temp/comp.jpg)     |
