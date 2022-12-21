import os
import sys
import json
import datetime
import numpy as np
from tqdm import tqdm
import joblib
# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import model as modellib, utils




############################################################
#  Configurations
############################################################


class DetectorConfig(Config):
    BACKBONE="resnet50"
    BATCH_SIZE=8
    DETECTION_MAX_INSTANCES=100
    DETECTION_MIN_CONFIDENCE=0
    DETECTION_NMS_THRESHOLD=0.3
    GPU_COUNT=1
    IMAGES_PER_GPU=8
    IMAGE_CHANNEL_COUNT=3
    IMAGE_MAX_DIM=256
    IMAGE_MIN_DIM=256
    IMAGE_MIN_SCALE=0
    LEARNING_MOMENTUM=0.9
    LEARNING_RATE=0.001
    LOSS_WEIGHTS={'rpn_class_loss': 1.0, 'rpn_bbox_loss': 1.0, 'mrcnn_class_loss': 1.0, 'mrcnn_bbox_loss': 1.0, 'mrcnn_mask_loss': 3.0}
    MASK_POOL_SIZE=14
    MASK_SHAPE=[28, 28]
    MAX_GT_INSTANCES=40
    MEAN_PIXEL=[50., 50., 50.]
    NAME="iris_feature_finetuned"
    NUM_CLASSES=2
    POST_NMS_ROIS_INFERENCE=1000
    POST_NMS_ROIS_TRAINING=2000
    RPN_ANCHOR_SCALES=(8, 16, 32, 64, 128)
    RPN_NMS_THRESHOLD=0.9
    RPN_TRAIN_ANCHORS_PER_IMAGE=256
    STEPS_PER_EPOCH=500
    TRAIN_ROIS_PER_IMAGE=256
    USE_MINI_MASK=False
    USE_RPN_ROIS=True
    WEIGHT_DECAY=0.01




############################################################
#  Dataset
############################################################

class DetectorDataset(utils.Dataset):
    def load_detector(self):
        # Add classes. We have only one class to add.
        self.add_class("iris_feature", 1, "iris_feature")
