
import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
import cv2 as cv
from copy import deepcopy
from tqdm import tqdm
import matplotlib.cm as cm
import shutil
import scipy.io

# Import Mask RCNN
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import detector
from tqdm import tqdm

from helper_methods import get_patch,get_centre,get_features,get_centre_fast,get_feature_masks
from run_matching import match_pair
from display_matching_patches import display_matching,draw_features_bsif

# %%
def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def run_clahe(img):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
    return img

def load_image(image_name,image_path,mask_path):
    # if not image_name.endswith(".png"):
    #     return

    image = cv.imread(os.path.join(image_path,image_name), cv.IMREAD_GRAYSCALE)
    mask = cv.imread(os.path.join(mask_path,image_name), cv.IMREAD_GRAYSCALE)

    masked_image = deepcopy(image).astype(np.uint8, copy=False)
    masked_image[mask == 0] = 0

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    masked_image = clahe.apply(masked_image)
    masked_image = cv.cvtColor(masked_image,cv.COLOR_GRAY2RGB)
    return [np.array(masked_image),mask,image]


def run_detection(image,model,DEVICE,w_ind,image_name):

    with tf.device(DEVICE):
        results = model.detect([image], verbose=0)
    r = results[0]
    masks = r['masks']
    N = r['rois'].shape[0]
    name= save_loc + w_ind + "/output/" + image_name + ".png"
    for i in range(N):
        if i < 10:
            plt.close('all')
            mask = masks[:, :, i]
            plt.imsave(name.replace(".png", "_mask_" + str(i) + ".png").replace("output","out_masks"), np.array(mask), cmap=cm.gray)

    # # Display results
    # ax = get_ax(1)
    
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                dataset.class_names, r['scores'], #ax=ax,
                                title="Predictions",name=name)
    plt.cla()


def run_bsif(image,filters,nm):
    codeImg = np.zeros(image.shape)
    num_filters = len(filters[0][0])
    for i in range(1,num_filters+1):
        ci = scipy.signal.convolve2d(image, np.rot90(filters[:,:,num_filters-i],2), mode='same', boundary='wrap')
        # cv.imwrite(nm + str(i) + ".png",255*(ci>0))
        codeImg=codeImg+(ci>0)*(2**(i-1))
    return codeImg.astype("uint8")


def run_inference(probe_image,gallery_image):

    if not os.path.exists(save_loc + str(w_ind) + "/output/"):
        os.makedirs(save_loc + str(w_ind) + "/output/")
    else:
        shutil.rmtree(save_loc + str(w_ind) + "/output/")
        os.makedirs(save_loc + str(w_ind) + "/output/")
    if not os.path.exists(save_loc + str(w_ind) + "/out_masks/"):
        os.makedirs(save_loc + str(w_ind) + "/out_masks/")
    else:
        shutil.rmtree(save_loc + str(w_ind) + "/out_masks/")
        os.makedirs(save_loc + str(w_ind) + "/out_masks/")

    probe = resized_images["probe"][0]
    gallery = resized_images["gallery"][0]
    probe_mask = resized_images["probe"][1]
    gallery_mask = resized_images["gallery"][1]
    probe_original = resized_images["probe"][2]
    gallery_original = resized_images["gallery"][2]

    probe_bsif = run_bsif(probe_original,filters,probe_image)
    gallery_bsif = run_bsif(gallery_original,filters,gallery_image)
    
    # Run object detection
    # cv.imwrite('./tester.png',probe)
    run_detection(probe,model,DEVICE,w_ind,"probe")
    run_detection(gallery,model,DEVICE,w_ind,"gallery")

    source_masks = save_loc + str(w_ind) + "/out_masks/"
    print("Loading Feature Masks from",source_masks,"...")
    feature_masks = get_feature_masks(source_masks)

    print("Generating Feature Representations for the patches...")
    probe_feats = get_features("probe",probe_bsif,feature_masks)
    gallery_feats = get_features("gallery",gallery_bsif,feature_masks)
    centre_probe = get_centre(probe_mask)
    centre_gallery = get_centre(gallery_mask)

    print("Matching Pair...")
    [matching_score,coord_pairings] = match_pair(probe_feats,gallery_feats,centre_probe,centre_gallery)

    classification_threshold = 1 - 0.6502472791056476
    if matching_score < classification_threshold:
        genuine = "Same Eye"
        classif = 1
    else:
        genuine = "Different Eyes"
        classif = 0

    print("Generating Visualization...")
    use_clahe_probe = cv.cvtColor(probe_original.copy(),cv.COLOR_GRAY2RGB)
    use_clahe_probe[probe_mask == 1] = probe[probe_mask == 1]
    use_clahe_gallery = cv.cvtColor(gallery_original.copy(),cv.COLOR_GRAY2RGB)
    use_clahe_gallery[gallery_mask == 1,:] = gallery[gallery_mask == 1,:]

    # display_matching(probe,gallery,coord_pairings,feature_masks,round(matching_score,4),genuine,run_clahe(probe_original),run_clahe(gallery_original))
    display_matching(probe,gallery,coord_pairings,feature_masks,round(matching_score,4),genuine,use_clahe_probe,use_clahe_gallery)
    return matching_score,classif
    


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run Inference on Mask R-CNN to detect iris features.')

    parser.add_argument('--out_dir', required=False, default="./stored_features/",
                        help='Directory to output detections to.')
    parser.add_argument('--model_path', required=False, default="./Model/wacv_model.h5",
                        help='Path to the model to use.')
    parser.add_argument('--textfile', required=False, default="example_pairs.txt",
                        help='Path to the text file of comparisons.')
    ## Configure path to cropped images and masks
    parser.add_argument('--cropped_image_path', required=False, default="./workdir/input/images/",
                        help='Path to cropped images.')
    parser.add_argument('--cropped_mask_path', required=False, default="./workdir/input/masks/",
                        help='Path to cropped masks.')
    ## Where to save visual to
    parser.add_argument('--destination', required=False, default="./workdir/patch-based/output/",
                        help='Path to save generated visualization to.')
    parser.add_argument('--scorefile', required=False, default="./workdir/patch-based/output/scores.txt",
                        help='Path to save generated scorefile.')
    args = parser.parse_args()

    before = time.time()


    # Load validation dataset
    dataset = detector.DetectorDataset()
    dataset.load_detector()

    # Must call before using the dataset
    dataset.prepare()

    config = detector.DetectorConfig()


    # Override the training configurations with a few
    # changes for inferencing.
    num_batch = 1
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = num_batch

        # RPN_NMS_THRESHOLD = 0.7
        
    config = InferenceConfig()
    config.display()

    # Device to load the neural network on.
    # Useful if you're training a model on the same 
    # machine, in which case use CPU and leave the
    # GPU for training.
    DEVICE = "gpu:0"  # /cpu:0 or /gpu:0

    # Inspect the model in training or inference modes
    # values: 'inference' or 'training'
    # TODO: code for 'training' test mode not ready yet
    TEST_MODE = "inference"
    weight_path = args.model_path
    w_ind = weight_path.split("/")[-1].replace('.h5','')

    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir="./", config=config,weight_dir="Model")
        # model = modellib.MaskRCNN(mode="inference", config=config)

    # Load weights
    print("Loading weights ", weight_path)
    model.load_weights(weight_path, by_name=True)
    print("Weights loaded")

    filters = scipy.io.loadmat('./BSIF_Filter/ICAtextureFilters_17x17_5bit.mat')['ICAtextureFilters']

    f = open(args.textfile,"r")
    out = open(args.scorefile,'w+')
    out.write('probe_image,gallery_image,genuine_pair,matching_distance\n')
    for comp in tqdm(f):
        tokens = comp.split(",")
        probe_image = tokens[0]
        gallery_image = tokens[1].replace("\n","")
        resized_images = {}
        resized_images["probe"] = load_image(probe_image,args.cropped_image_path,args.cropped_mask_path)
        resized_images["gallery"] = load_image(gallery_image,args.cropped_image_path,args.cropped_mask_path)

        # %%
        # Set path to iris weights file
        save_loc = args.out_dir
        print(save_loc)
        match_score,genuine = run_inference(probe_image,gallery_image)
        total_runtime = time.time() - before
        print("Time taken to match pair of images:",total_runtime)
        pair_name = probe_image.replace(".png","") + "-" + gallery_image 
        plt.tight_layout(pad=0)
        plt.savefig(args.destination + pair_name)
        # plt.show()
        plt.cla()
        plt.close()
        out.write(probe_image + "," + gallery_image + "," + str(genuine) + "," + str(match_score) + "\n")


