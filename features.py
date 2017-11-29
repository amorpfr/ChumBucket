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

def extract_features(image):
    # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(image)

    # take the mean of it and return it
    ht_mean = textures.mean(axis=0)
    return ht_mean
    
if __name__ == "__main__":
    path = 'images.pkl'
    images  = pd.read_pickle(path)
    images['clean'] = images['image_matrix'].apply(thresholding)
    image = images['image_matrix'].iloc[0] 
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    features = extract_features(image)