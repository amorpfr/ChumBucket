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

if __name__ == "__main__":
    path = 'C:/Users/daniel/Documents/AML Kaggle/images.pkl'
    images  = pd.read_pickle(path)
    images['mean'] = images['image_matrix'].apply(thresholding)
    
    plt.figure(figsize = (12,3))
    sub = plt.subplot(1,4,1)
    plt.imshow(images['mean'].loc[1], cmap='gray')