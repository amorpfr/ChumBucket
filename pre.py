import pandas as pd
import pprint as pp
import numpy as np
import cv2
from skimage import morphology
from skimage import measure
from skimage.transform import resize

def to_image(image_name):
#    path = './data/train_images/' + image_name
    path = 'C:/Users/daniel/Documents/AML Kaggle/train_images/' + image_name
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

def resize_image(image):
    """
    Program: Resizes image based on global variable maxPixel
    Input: image
    Output: Resized(Squared) image
    """
    maxPixel = 20
    image = resize(image, (maxPixel, maxPixel), preserve_range= True)
    return image

def pre_haralick(image):
    
    maxregion = get_important_region(image)
    
    small_img = image[maxregion.bbox[0]:maxregion.bbox[2],maxregion.bbox[1]:maxregion.bbox[3]]
    noNoise = np.where(small_img > np.mean(image),0.,1.0)
    
    pre_haralick = small_img * noNoise
    
    return pre_haralick

def superb(image):
    
    maxregion = get_important_region(image)
    
    small_img = image[maxregion.bbox[0]:maxregion.bbox[2],maxregion.bbox[1]:maxregion.bbox[3]]
    noNoise = np.where(small_img > np.mean(image),0.,1.0)
    plusje = np.where(small_img > np.mean(image),255.,0.)
    perfect = (small_img * noNoise)+plusje
    
    superb = resize_image(perfect)
#    superb = np.pad(klein, (5,5),'constant')
#    superb[superb == 0] = 255    
#    superb = np.where(superb > 250,0.,1.0)
    return superb


if __name__ == "__main__":
    images  = pd.read_csv("./data/train_onelabel.csv", encoding='utf-8')
    images['image_matrix'] = images['image'].apply(to_image)
    images['threshold'] = images['image_matrix'].apply(thresholding)
    images['clean'] = images['image_matrix'].apply(clean)
    images['superb'] = images['image_matrix'].apply(superb)
    images['pre_haralick'] = images['image_matrix'].apply(pre_haralick)
    end_df = images.sample(500)
    #end_df = images[:100]   
    end_df.to_pickle("images500.pkl")


