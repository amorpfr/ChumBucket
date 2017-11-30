import pandas as pd
import pprint as pp
import numpy as np
import cv2
from skimage import morphology
from skimage import measure

def to_image(image_name):
    path = './data/train_images/' + image_name
    img = cv2.imread(path,0)
    return img


def thresholding(image):
    """
    Program: Thresholding image following the tutorial
    Input: Image
    Output: Cleaned image
    """
    imthr = np.where(image > np.mean(image),0.,1.0)
    imdilated = morphology.dilation(imthr, np.ones((4,4)))
    labels = measure.label(imdilated)
    labels = imthr * labels
    labels = labels.astype(int)
    return labels

def clean(image):
    """
    Program: Cleaning image without diliated
    Input: Image
    Output: Cleaned image
    """
    imthr = np.where(image > np.mean(image),0.,1.0)
    #plusje = np.where(image > np.mean(image),255.,0.)
    cleaned = image * imthr
    #cleaned = cleaned + plusje
    cleaned = cleaned.astype(int)
    return cleaned


if __name__ == "__main__":
    images  = pd.read_csv("./data/train_onelabel.csv", encoding='utf-8')
    images['image_matrix'] = images['image'].apply(to_image)
    images['threshold'] = images['image_matrix'].apply(thresholding)
    images['clean'] = images['image_matrix'].apply(clean)
    end_df = images.sample(1000)
    #end_df = images[:100]   
    end_df.to_pickle("images1000.pkl")


