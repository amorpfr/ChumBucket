from skimage import morphology
from skimage import measure
import pandas as pd
import pprint as pp
import numpy as np
import matplotlib.pyplot as plt
import cv2

def thresholding(image):
    imthr = np.where(image > np.mean(image),0.,1.0)
    imdilated = morphology.dilation(imthr, np.ones((4,4)))
    labels = measure.label(imdilated)
    labels = imthr * labels
    labels = labels.astype(int)
    return labels

def original(image):
    imthr = np.where(image > np.mean(image),0.,1.0)
    imdilated = morphology.dilation(imthr, np.ones((4,4)))
    gray_image = image / np.max(image)
    cleaned = imdilated * gray_image
    return cleaned

if __name__ == "__main__":
    path = 'C:/Users/daniel/Documents/AML Kaggle/images.pkl'
    images  = pd.read_pickle(path)
#    images['threshold'] = images['image_matrix'].apply(thresholding)
    images['clean'] = images['image_matrix'].apply(original)
#    plt.figure(figsize = (12,3))
#    sub = plt.subplot(1,4,1)
    plt.imshow(images['clean'].loc[1], cmap='gray')

#    for item in images.clean.loc[[0]][0]:
#        print item
        
    