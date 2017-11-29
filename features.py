from skimage import morphology
from skimage import measure
import pandas as pd
import pprint as pp
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mahotas as mt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import itertools


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
    #cleaned = cleaned + plusje
    cleaned = cleaned.astype(int)
    return cleaned
    
    
def haralick(image):
    # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(image, ignore_zeros= True)

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
    return np.array([ratio])  

def getSize(image):
    shape = np.shape(image)
    return shape

def merge_features(featureset):
    tmp = featureset.iloc[0].tolist()
    tmp1 = merge(tmp)
    feature_matrix = np.zeros((len(featureset), len(tmp1)))
    i=0
    for i in range(0, len(featureset)):
        #print(i/len(featureset))
        feature_matrix[i] = merge(featureset.iloc[i].tolist())
    return feature_matrix    
    
def merge(row):
    merged = np.concatenate(row)
    return merged

def test(images):
    features = images[['haralick','ratio']]
    features = merge_features(features)
    labels = images[['class']]
    trainX, testX, trainY, y_true = train_test_split(features, labels, test_size=0.2, random_state=42)
    clf = SGDClassifier(loss="hinge", penalty="l2")
    clf.fit(trainX, trainY)
    y_pred = clf.predict(testX)
    accuracy = float(accuracy_score(y_true, y_pred, normalize =True))
    return accuracy

if __name__ == "__main__":
    # load files
    path = 'images_test_mixed.pkl'
    images  = pd.read_pickle(path)
    print("Images loaded")
    
    # clean images
    images['threshold'] = images['image_matrix'].apply(thresholding)
    images['clean'] = images['image_matrix'].apply(clean)
    print("Images cleaned")
    
    # extract features
    images['haralick'] = images['clean'].apply(haralick)
    images['ratio'] = images['clean'].apply(getMinorMajorRatio)
    print("Features extracted")
   
    # test model
    accuracy = test(images)
    print("Training done, testing accuracy: ", accuracy)
    print("") 
    
    