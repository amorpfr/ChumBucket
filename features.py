from skimage import morphology
from skimage import measure
import pandas as pd
import pprint as pp
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mahotas as mt
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.metrics import log_loss
from skimage.transform import resize
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.ensemble import RandomForestClassifier as RF
    
def haralick(image):
    """
    Program: calculate haralick texture features for 4 types of adjacency
    Input: Image
    Output: 13 haralick features (numpy array)
    """
    image = image.copy()
    image= image.astype(int)
#    textures = mt.features.haralick(image)
    textures = mt.features.haralick(image, ignore_zeros= True)
    
    # take the mean of it and return it
    ht_mean = textures.mean(axis=0)
    return ht_mean

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

def merge_features(featureset):
    """
    Program: Merge features in one feature matrix
    Input: Dataframe of features
    Output: Numpy matrix of merged features
    """
    tmp = featureset.iloc[0].tolist()
    tmp1 = merge(tmp)
    feature_matrix = np.zeros((len(featureset), len(tmp1)))
    i=0
    for i in range(0, len(featureset)):
        #print(i/len(featureset))
        feature_matrix[i] = merge(featureset.iloc[i].tolist())
    return feature_matrix    
    
def merge(row):
    """
    Program: Merge multiple numpy arrays in one numpy array
    Input: List of numpy arrays
    Output: One numpy array of the merged arrays
    """
    merged = np.concatenate(row)
    return merged

def pixel_feature(image):
    """
    Program: Flattens the image to create features of the pixels
    Input: image
    Output: 1-d numpy array of pixels
    """
    pixels = image.flatten()
    return pixels

def zernike(image):
    """
    Program: Computes the zernike moments around the center of mass in the image
    Input: pre_zernike image
    Output: 25 dimensions of zernike moments.
    """
    
    zernike_features = mt.features.zernike_moments(image, radius=15, degree=8)
    return zernike_features

def linear_binary_pattern(image):
    lbp =   mt.features.lbp(image, radius=4, points=4, ignore_zeros=False)
    # 12 points werkt beter, maar veel meer computing time
    return lbp

def linear_binary_pattern2(image):
    lbp =   mt.features.lbp(image, radius=8, points=8, ignore_zeros=False)
    # 12 points werkt beter, maar veel meer computing time
    return lbp

def linear_binary_pattern3(image):
    lbp =   mt.features.lbp(image, radius=8, points=12, ignore_zeros=False)
    # 12 points werkt beter, maar veel meer computing time
    return lbp

def test(images, used_features):
    """
    Program: Computes the accuracy of a dataset consisisting images, features and classes
    Input: Dataframe consisting all data, list of names of features to be used
    Output: Accuracy
    """
    features = images[used_features]
    features = merge_features(features)
    labels = np.array(images['class'])
    
    clf = RF(n_estimators=100, n_jobs=3);
    scores = cross_validation.cross_val_score(clf, features, labels, cv=5, n_jobs=1);
    accuracy = np.mean(scores)

    return accuracy


if __name__ == "__main__":
    
    start_time = time.time()

    # load files
    print('Loading images ...')
    path = 'images_test.pkl'
    images  = pd.read_pickle(path)
    print("Images loaded")
    
    # extract features
    print('Extracting features ...')

    images['ratio'] = images['image_matrix'].apply(getMinorMajorRatio)
#    images['pixels'] = images['superb'].apply(pixel_feature)
    images['image_size'] = images['pre_haralick'].apply(getSize)
    print("Feature size extracted")
    images['haralick'] = images['pre_haralick'].apply(haralick)
    print("Feature haralick extracted")
    images['zernike'] = images['pre_zernike'].apply(zernike)
    print("Feature zernike extracted")
    images['binary_pattern_small'] = images['pre_haralick'].apply(linear_binary_pattern) 
    print("Feature binary pattern small extracted")
    images['binary_pattern'] = images['pre_haralick'].apply(linear_binary_pattern2) 
#    images['binary_pattern_big'] = images['threshold'].apply(linear_binary_pattern3)    
    print("Features extracted")
   
    # test model
    print('Training model ...')
#    features_to_use = ['ratio','image_size','haralick','zernike','binary_pattern', 'binary_pattern_small']
#    accuracy = test(images, features_to_use)
    print("")
#    print("Training done, testing accuracy: ", accuracy)
    print("") 
    
    # Create pickle
    print('Creating pickle ...')
    del images['image_matrix']
    del images['pre_zernike']
    del images['pre_haralick']
    images.to_pickle('features_test.pkl')
    print('Pickle created')
    
    print("--- %s seconds ---" % (time.time() - start_time))
