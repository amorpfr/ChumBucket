"""
Retrieves different global and local features from pre-processed plankton
images and saves them to a dataframe.
"""
from skimage import morphology
from skimage import measure
import pandas as pd
import numpy as np
import mahotas as mt
import time


def getLargestRegion(props, labelmap, imagethres):
    """
    Program: Gets most significant region(>50% nonzero)
    Input: Image
    Output: Region
    """
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
    """
    Program: Computes ratio of the most significant region in image
    Input: Image
    Output: Ratio of the most significant region (numpy array)
    """
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
    return np.array([ratio])  

def get_important_region(image):
    """
    Program: Retrieves upper and lower boundary of height and width of
    largest object in image
    Input: Image
    Output: Coordinates of object outline(numpy array)
    """
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

    return maxregion

def getSize(image):
    """
    Program: Get size of image
    Input: image
    Output: width and length in numpy array
    """
    (x,y) = np.shape(image)
    return np.array([x,y])

def pixel_feature(image):
    """
    Program: Flattens the image to create features of the pixels
    Input: image
    Output: 1-d numpy array of pixels
    """
    pixels = image.flatten()
    return pixels

def haralick(image):
    """
    Program: calculate haralick texture features for 4 types of adjacency
    Input: Image
    Output: 13 haralick features (numpy array)
    """
    image = image.copy()
    image= image.astype(int)
    textures = mt.features.haralick(image, ignore_zeros= True)
    
    ht_mean = textures.mean(axis=0)
    return ht_mean

def zernike(image):
    """
    Program: Computes the zernike moments around the center of mass in the image
    Input: pre_zernike image
    Output: 25 dimensions of zernike moments.
    """
    
    zernike_features = mt.features.zernike_moments(image, radius=15, degree=8)
    return zernike_features

def linear_binary_pattern(image):
    """
    Program: Computes the binary code for each pixel using a certain radius
    around that pixel and a certain amount of evenly-spaced points on that radius.
    Input: pre_haralick image
    Output: Histogram of binary codes.
    """
    lbp =   mt.features.lbp(image, radius=4, points=4, ignore_zeros=False)
    return lbp

def linear_binary_pattern2(image):
    """
    Program: Same as the function above but with a different radius and a
    different amount of points.
    """
    lbp =   mt.features.lbp(image, radius=8, points=8, ignore_zeros=False)
    return lbp

def linear_binary_pattern3(image):
    """
    Program: Same as the function above but with a different radius and a
    different amount of points.
    """
    lbp =   mt.features.lbp(image, radius=8, points=12, ignore_zeros=False)
    return lbp

if __name__ == "__main__":
    
    start_time = time.time()

    # load files
    print('Loading images ...')
    path = 'preprocessed.pkl'
    images  = pd.read_pickle(path)
    print("Images loaded")
    
    # extract features
    print('Extracting features ...')

    images['ratio'] = images['image_matrix'].apply(getMinorMajorRatio)
    images['pixels'] = images['superb'].apply(pixel_feature)
    images['image_size'] = images['pre_haralick'].apply(getSize)
    print("Feature size extracted")
    images['haralick'] = images['pre_haralick'].apply(haralick)
    print("Feature haralick extracted")
    images['zernike'] = images['pre_zernike'].apply(zernike)
    print("Feature zernike extracted")
    images['binary_pattern_small'] = images['pre_haralick'].apply(linear_binary_pattern) 
    print("Feature binary pattern small extracted")
    images['binary_pattern'] = images['pre_haralick'].apply(linear_binary_pattern2) 
    images['binary_pattern_big'] = images['threshold'].apply(linear_binary_pattern3)    
    print("Features extracted")
   
    
    # Create pickle
    print('Creating pickle ...')
    del images['image_matrix']
    del images['pre_zernike']
    del images['pre_haralick']
    images.to_pickle('features.pkl')
    print('Pickle created')

    print("--- %s seconds ---" % (time.time() - start_time))
