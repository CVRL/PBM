import os

import numpy as np
from imutils import paths
import cv2 as cv
import numpy.random as rng
import sys
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from multiprocessing import Pool
import matplotlib.pyplot as plt
import json
import shapely.geometry
from scipy.spatial import distance
import random
from operator import itemgetter 
import time
from sklearn import multiclass
from sklearn import metrics
import warnings
import itertools

np.seterr(all='warn')


def get_features(im_name,image,feature_masks):
    min_feature_size = 10
    im_feats = []
    if im_name.replace(".png","") in feature_masks:
        image_features = feature_masks[im_name.replace(".png","")]
        # print(im_name)
        for mask in image_features:
            [centre_point_x,centre_point_y,patch_dims] = get_centre(mask)
            # get_patch(coords,image,patch_size)
            
            if centre_point_x > 256 or centre_point_y > 256 or centre_point_x == 0 or centre_point_y == 0:
                continue
            image_patch = get_patch([centre_point_x,centre_point_y],image,patch_dims)

            if patch_dims[0] < min_feature_size or patch_dims[1] < min_feature_size:
                continue
            # mask_patch = get_patch([centre_point_x,centre_point_y],mask,patch_dims)
            max_bits = len(bin(np.max(image_patch))[2:])
            feats = np.zeros((patch_dims[1],patch_dims[0],max_bits))
            for idy,row in enumerate(image_patch):
                for idx,val in enumerate(row):
                    binary = bin(val)[2:]
                    reverse_binary = binary[::-1]
                    for idz,bit in enumerate(reverse_binary):
                        feats[idy,idx,idz] = int(bit)
            
            im_feats.append([feats,[centre_point_x,centre_point_y]])
    return im_feats

def get_patch(coords,image,patch_size):
    h,w = image.shape
    p_width = patch_size[0]
    half_w = int(p_width/2)
    p_height = patch_size[1]
    half_h = int(p_height/2)

    x_coord = round(coords[0])
    y_coord = round(coords[1])

    if x_coord - half_w < 0:
        x_coord = half_w
    elif x_coord + half_w > w:
        x_coord = w - half_w

    if y_coord - half_h < 0:
        y_coord = half_h
    elif y_coord + half_h > h:
        y_coord = h - half_h

    patch = image[y_coord - half_h:y_coord + half_h, x_coord - half_w:x_coord + half_w]

    return patch


def get_feature_masks(source_masks_feat): # 0102_L_4_1_mask_1.png
    feature_masks = {}
    for feature_mask in tqdm(os.listdir(source_masks_feat)):
        mask = cv.imread(source_masks_feat + "/" + feature_mask, cv.IMREAD_GRAYSCALE)
        im_name = feature_mask[:feature_mask.index("_mask_")]
        if im_name in feature_masks:
            feature_masks[im_name].append(mask)
        else:
            feature_masks[im_name] = []
            feature_masks[im_name].append(mask)
    return feature_masks


def get_centre(mask):
    min_x = 100000
    max_x = 0
    min_y = 100000
    max_y = 0
    for idy,row in enumerate(mask):
        if 255 in row:
            if idy < min_y:
                min_y = idy
            elif idy > max_y:
                max_y = idy
            for idx, col in enumerate(row):
                if col == 255:
                    if idx < min_x:
                        min_x = idx
                    elif idx > max_x:
                        max_x = idx
    centre_point_x = round((max_x - min_x)/2) + min_x
    centre_point_y = round((max_y - min_y)/2) + min_y

    width = max_x - min_x
    height = max_y - min_y

    dimensions = [width,height]
    # print(dimensions)

    # plt.imshow(mask)
    # plt.show()
    return [centre_point_x,centre_point_y,dimensions]

def get_centre_fast(mask):
    M = cv.moments(mask)
    # calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX = 0
        cY = 0
    # if cX > 256 or cY > 256:
    #     print("Big error")
    #     print(cX,cY)
    return [cX,cY]

def run_matching(features1,features2,centre_mask1,delta_x,delta_y,angle_thresh=20):
    # warnings.filterwarnings('error')
    
    pairings = {}
    all_patch_scores = {}
    all_patch_pairs = {}
    coord_pairings = []
    # print("start")
    # p1_xs = []
    # p1_ys = []
    for idx, patch1 in enumerate(features1):
        patch1_feats = patch1[0]
        patch1_coords = patch1[1]

        for idy, patch2 in enumerate(features2):

            patch2_feats = patch2[0]
            patch2_coords = patch2[1]

            patch2_coords_corrected = [patch2_coords[0] + delta_x, patch2_coords[1] + delta_y]

            ba = np.array(patch1_coords) - np.array(centre_mask1)
            bc = np.array(patch2_coords_corrected) - np.array(centre_mask1)
            

            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                except:
                    cosine_angle = 420
                    # continue
                if cosine_angle == 420:
                    print("BEEEEEEP")
                if cosine_angle < -1 or cosine_angle > 1:
                    # print(cosine_angle)
                    # angle = np.degrees(np.arccos(cosine_angle))
                    # print(angle)
                    if cosine_angle < -1:
                        cosine_angle = -1
                    elif cosine_angle > 1:
                        cosine_angle = 1
                    # angle = np.degrees(np.arccos(cosine_angle))
                    # print(angle)
                try:
                    angle = np.degrees(np.arccos(cosine_angle))
                except:
                    print(cosine_angle)

            if abs(angle) >= angle_thresh:
                continue
            name_pair = str(idx) + "-" + str(idy)
            dist = get_distance(patch1_feats,patch2_feats)
            # dist = 0.8
            # dist = np.dot(patch1_feats, patch2_feats)/(np.linalg.norm(patch1_feats)*np.linalg.norm(patch2_feats))
            # print(di  st)
            all_patch_scores[name_pair] = dist# * angle
            all_patch_pairs[name_pair] = [patch1_coords,patch2_coords]
    patch1_used = []
    patch2_used = []
    for key, value in sorted(all_patch_scores.items(), key = itemgetter(1), reverse = False):
        tkns = key.split("-")
        idx = tkns[0]
        if idx not in patch1_used:
            idy = tkns[1]
            if idy not in patch2_used:
                patch1_used.append(idx)
                patch2_used.append(idy)
                pairings[str(idx) + "-" + str(idy)] = value
                coord_pairings.append(all_patch_pairs[str(idx) + "-" + str(idy)])
    # print(coord_pairings)
    return [find_score(pairings),pairings,coord_pairings]
    # return [[find_score(pairings),find_score2(pairings),find_score3(pairings),find_score4(pairings),find_score5(pairings),find_score6(pairings),find_score7(pairings),find_score8(pairings)],pairings]


def hamming_3d(p1, p2):
   ## assuming p1 and p2 are same shape of [x,y,filters]
    n = p1.shape[0] * p1.shape[1]
    diff = p1 != p2
    #    print(p1.shape,p2.shape)
    #    print(diff.shape,n)
    try:
        dists = np.sum(diff, axis=(0,1)) / n  #This should be of shape [filters]
    except:
        dists = [1]
   
    return dists


def get_distance(patch1,patch2):
    p1_h,p1_w,p1_z = patch1.shape
    p2_h,p2_w,p2_z = patch2.shape
    min_z = min(p1_z,p2_z)
    min_distance = 10000

    area_p1 = p1_h * p1_w
    area_p2 = p2_h * p2_w
    smaller_area = np.min([area_p1,area_p2])


    if p1_h >= p2_h and p1_w >= p2_w: # Patch 1 bigger than or equal to patch 2
        diff_h = p1_h - p2_h
        diff_w = p1_w - p2_w

        all_shifts = list(itertools.product(list(range(diff_h+1)),list(range(diff_w+1))))
        for shift in all_shifts:
            y_shift = shift[0]
            x_shift = shift[1]
            reduced_p1 = patch1[y_shift:y_shift + p2_h,x_shift:x_shift+p2_w,:min_z]
            overlap_area = reduced_p1.shape[0]*reduced_p1.shape[1]
            if (2*overlap_area<=smaller_area):
                return 1.0
            all_mins = hamming_3d(reduced_p1,patch2)
            checker_dist = np.mean(all_mins)

            if checker_dist < min_distance:
                min_distance = checker_dist

    elif p1_h <= p2_h and p1_w <= p2_w: # Patch 2 bigger than 1
        diff_h = p2_h - p1_h
        diff_w = p2_w - p1_w

        all_shifts = list(itertools.product(list(range(diff_h+1)),list(range(diff_w+1))))
        for shift in all_shifts:
            y_shift = shift[0]
            x_shift = shift[1]
            reduced_p2 = patch2[y_shift:y_shift + p1_h,x_shift:x_shift+p1_w,:min_z]
            overlap_area = reduced_p2.shape[0]*reduced_p2.shape[1]
            if (2*overlap_area<=smaller_area):
                return 1.0
            # mean_hd = []
            all_mins = hamming_3d(patch1,reduced_p2)
            checker_dist = np.mean(all_mins)
            if checker_dist < min_distance:
                # min_distance = np.mean(mean_hd)
                min_distance = checker_dist
    elif p1_h >= p2_h and p1_w <= p2_w: # Patch 1 taller but patch 2 wider
        diff_h = p1_h - p2_h
        diff_w = p2_w - p1_w

        all_shifts = list(itertools.product(list(range(diff_h+1)),list(range(diff_w+1))))
        for shift in all_shifts:
            y_shift = shift[0]
            x_shift = shift[1]
            # mean_hd = []
            reduced_p1 = patch1[y_shift:y_shift + p2_h,:,:min_z]
            reduced_p2 = patch2[:,x_shift:x_shift+p1_w,:min_z]
            overlap_area = reduced_p1.shape[0]*reduced_p1.shape[1]
            if (2*overlap_area<=smaller_area):
                return 1.0
            all_mins = hamming_3d(reduced_p1,reduced_p2)
            checker_dist = np.mean(all_mins)
            if checker_dist < min_distance:
                min_distance = checker_dist
    elif p1_h <= p2_h and p1_w >= p2_w: # Patch 2 taller but patch 1 wider
        diff_h = p2_h - p1_h
        diff_w = p1_w - p2_w

        all_shifts = list(itertools.product(list(range(diff_h+1)),list(range(diff_w+1))))
        for shift in all_shifts:
            y_shift = shift[0]
            x_shift = shift[1]
            # mean_hd = []
            reduced_p1 = patch1[:,x_shift:x_shift+p2_w,:min_z]
            reduced_p2 = patch2[y_shift:y_shift + p1_h,:,:min_z]
            overlap_area = reduced_p1.shape[0]*reduced_p1.shape[1]
            if (2*overlap_area<=smaller_area):
                return 1.0
            all_mins = hamming_3d(reduced_p1,reduced_p2)
            checker_dist = np.mean(all_mins)
            if checker_dist < min_distance:
                min_distance = checker_dist
    else:
        print("Check this case boi")
    return min_distance

def find_score(pairings):
    k = [float("inf"), float("inf"), float("inf"), float("inf"), float("inf")]#, float("inf"), float("inf"), float("inf"), float("inf"), float("inf")]
    len_k = len(k)
    for pairing in pairings:

        score_pair = pairings[pairing]
        if score_pair < k[0]:
            k[0] = score_pair
            k.sort(reverse=True)
    k = list(filter(lambda a: a != float("inf"), k))
    if k == []: 
        matching_score = None
        # print("Empty")
    else:
        matching_score = np.mean(k)
    # print(matching_score)
    return matching_score

