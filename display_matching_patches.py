import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import cm

alpha = 0.2

def draw_features_full(img1,feature_masks):
    for mask in feature_masks:
        
        # print(mask.shape)
        r_channel = mask.copy()
        r_channel[mask != 0] = 0
        g_channel = mask.copy()
        g_channel[mask != 0] = 255
        b_channel = mask.copy()
        b_channel[mask != 0] = 255
        mask = cv.cvtColor(mask,cv.COLOR_GRAY2RGB)
        mask[:,:,0] = r_channel
        mask[:,:,1] = g_channel
        mask[:,:,2] = b_channel

        cv.addWeighted(mask, alpha, img1, 1,0, img1)
    return img1

def draw_features(img1,feature_masks):
    for mask in feature_masks:

        cont = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(img1, cont[0], -1, (0, 255, 255))
        
        # # print(mask.shape)
        # r_channel = mask.copy()
        # r_channel[mask != 0] = 0
        # g_channel = mask.copy()
        # g_channel[mask != 0] = 255
        # b_channel = mask.copy()
        # b_channel[mask != 0] = 255
        # mask = cv.cvtColor(mask,cv.COLOR_GRAY2RGB)
        # mask[:,:,0] = r_channel
        # mask[:,:,1] = g_channel
        # mask[:,:,2] = b_channel

        # cv.addWeighted(mask, alpha, img1, 1,0, img1)
    return img1
    

def draw_features_bsif(img1,feature_masks):
    zeros = np.zeros((img1.shape))
    for mask in feature_masks:

        zeros[mask==255,:] = img1[mask==255,:]

        cont = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(zeros, cont[0], -1, (255, 162, 0),2)
        
        # # print(mask.shape)
        # r_channel = mask.copy()
        # r_channel[mask != 0] = 0
        # g_channel = mask.copy()
        # g_channel[mask != 0] = 255
        # b_channel = mask.copy()
        # b_channel[mask != 0] = 255
        # mask = cv.cvtColor(mask,cv.COLOR_GRAY2RGB)
        # mask[:,:,0] = r_channel
        # mask[:,:,1] = g_channel
        # mask[:,:,2] = b_channel

        # cv.addWeighted(mask, alpha, img1, 1,0, img1)
    return zeros,img1

def display_matching(probe,gallery,matches,feature_masks,matching_score,classification,original_probe,original_gallery):

    img1 = probe
    img2 = gallery

    img1 = draw_features(img1,feature_masks["probe"])
    img2 = draw_features(img2,feature_masks["gallery"])

    comb_image = np.concatenate((img1,img2),axis=1)
    # overlay = comb_image.copy()
    for coord_pair in matches:
        # print(coord_pair)
        coord_1 = coord_pair[0]
        coord_2 = coord_pair[1]

        line_thickness = 2
        # cv.line(overlay, (coord_1[0], coord_1[1]), (coord_2[0]+256, coord_2[1]), (0, 0, 255), thickness=line_thickness)
        cv.line(comb_image, (coord_1[0], coord_1[1]), (coord_2[0]+256, coord_2[1]), (0, 0, 255), thickness=line_thickness)
    
    # cv.addWeighted(overlay, 0.6, comb_image, 1,0, comb_image)
    comb_original = np.concatenate((original_probe,original_gallery),axis=1)

    comb_both = np.concatenate((comb_original,comb_image),axis=0)
    # comb_both = comb_image
    
    plt.title("Matching Score: " + str(matching_score) + ", Classification: " + classification)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    frame1.imshow(comb_both)
    




