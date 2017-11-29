from skimage import morphology
from skimage import measure
import pandas as pd
import pprint as pp
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mahotas as mt


def thresholding(image):
    imthr = np.where(image > np.mean(image),0.,1.0)
    imdilated = morphology.dilation(imthr, np.ones((4,4)))
    labels = measure.label(imdilated)
    labels = imthr * labels
    labels = labels.astype(int)
    return labels

def clean(image):
    imthr = np.where(image > np.mean(image),0.,1.0)
    plusje = np.where(image > np.mean(image),255.,0.)
    cleaned = image * imthr
    cleaned = cleaned + plusje
    cleaned = cleaned.astype(int)
    return cleaned
    
    
def haralick(image):
    # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(image)

    # take the mean of it and return it
    ht_mean = textures.mean(axis=0)
    return ht_mean

def getLargestRegion(props, labelmap, imagethres):
    regionmaxprop = None
    for regionprop in props:
        # check to see if the region is at least 50% nonzero
        if sum(imagethres[labelmap == regionprop.label])*1.0/regionprop.area < 0.50:
            continue
        if regionmaxprop is None:
            regionmaxprop = regionprop
        if regionmaxprop.filled_area < regionprop.filled_area:
            regionmaxprop = regionprop
    return regionmaxprop

def getMinorMajorRatio(image):
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    imagethr = np.where(image > np.mean(image),0.,1.0)

    #Dilate the image
    imdilated = morphology.dilation(imagethr, np.ones((4,4)))

    # Create the label list
    label_list = measure.label(imdilated)
    label_list = imagethr*label_list
    label_list = label_list.astype(int)
    
    region_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(region_list, label_list, imagethr)
    
    # guard against cases where the segmentation fails by providing zeros
    ratio = 0.0
    if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
        ratio = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0 / maxregion.major_axis_length
    return ratio   

def getSize(image):
    shape = np.shape(image)
    return shape

if __name__ == "__main__":
    path = 'images_test.pkl'
    images  = pd.read_pickle(path)
    images['threshold'] = images['image_matrix'].apply(thresholding)
    images['clean'] = images['image_matrix'].apply(clean)
    image_dan = images['clean'].iloc[0] 
    image_tut = images['threshold'].iloc[0] 
    image_ori = images['image_matrix'].iloc[0] 
    
    plt.imshow(image_tut)
    features_tut = {}
    features_tut['haralick'] = haralick(image_tut)
    features_tut['ratio'] = getMinorMajorRatio(image_tut)
    
    plt.imshow(image_dan)
    features_dan = {}
    features_dan['haralick'] = haralick(image_dan)
    features_dan['ratio'] = getMinorMajorRatio(image_dan)
    
    plt.imshow(image_ori)
    features_ori = {}
    features_ori['haralick'] = haralick(image_ori)
    features_ori['ratio'] = getMinorMajorRatio(image_ori)
    
    