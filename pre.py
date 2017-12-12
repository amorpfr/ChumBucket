"""
This script pre-processes the images, cleaning them and computing feature-
specific images.
"""
import pandas as pd
import numpy as np
import cv2
from skimage import morphology
from skimage import measure
from skimage.transform import resize

def to_image(image_name):
    """
    Program: Converts image path to image 
    Input: Image name
    Output: Image as numpy matrix
    """    
    if test:
        path = './data/test_images/' + image_name
    else:
        path = './data/train_images/' + image_name
    img = cv2.imread(path,0)
    return img

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

def resize_image(image, resWidth, resHeight):
    """
    Program: Resizes image based on height and width input
    Input: image
    Output: Resized(Squared) image
    """
    image = resize(image, (resWidth, resHeight), preserve_range= True, mode='wrap')
    return image

def pre_haralick(image):
    """
    Program: Thresholds image, retrieves largest object and set background value to 0
    Input: image
    Output: Prepared image to retrieve haralick features
    """
    maxregion = get_important_region(image)
    
    small_img = image[maxregion.bbox[0]:maxregion.bbox[2],maxregion.bbox[1]:maxregion.bbox[3]]
    noNoise = np.where(small_img > np.mean(image),0.,1.0)
    
    pre_haralick = small_img * noNoise
    
    return pre_haralick

def pre_zernike(image):
    """
    Program: Thresholds image, retrieves largest object and resizes image
    Input: image
    Output: Prepared image to retrieve zernike features
    """
    
    maxregion = get_important_region(image)
    
    small_img = image[maxregion.bbox[0]:maxregion.bbox[2],maxregion.bbox[1]:maxregion.bbox[3]]
    noNoise = np.where(small_img > np.mean(image),0.,1.0)
    plusje = np.where(small_img > np.mean(image),255.,0.)
    perfect = (small_img * noNoise)+plusje
    
    resWidth = 20
    resHeight = 20
    pre_zernike = resize_image(perfect, resWidth, resHeight)

    return pre_zernike

def pre_surf(image):
    """
    Program: Threshold image, retrieves largest object, subtracts minimum and 
    resizes image
    Input: image
    Output: Prepared image to retrieve SURF features
    """
    maxregion = get_important_region(image)
    
    small_img = image[maxregion.bbox[0]:maxregion.bbox[2],maxregion.bbox[1]:maxregion.bbox[3]]
    noNoise = np.where(small_img > np.mean(image),0.,1.0)
    plusje = np.where(small_img > np.mean(image),255.,0.)
    minimum = np.min(small_img)
    
    cleaned = (small_img * noNoise) + plusje - minimum

    resWidth = 100
    resHeight = np.floor(resWidth / (float(image.shape[0])/image.shape[1]))
    
    pre_surf = resize_image(cleaned, resWidth, resHeight)
    
    return pre_surf


if __name__ == "__main__":
    
    # load files
    print('Loading images ...')
    
    # if run on test-set set test =True
    test = False
    
    # train images
    images  = pd.read_csv("./data/train_onelabel.csv", encoding='utf-8')
    
    # test images
    #images  = pd.read_csv("./data/sample.csv", encoding='utf-8')
    
    images['image_matrix'] = images['image'].apply(to_image)
    print("Images loaded")
    
    # Preprocessing images
    print('Pre-process images ...')
    images['pre_zernike'] = images['image_matrix'].apply(pre_zernike)
    images['pre_haralick'] = images['image_matrix'].apply(pre_haralick)
    images['pre_surf1'] = images['image_matrix'].apply(pre_surf)
    print("Images pre-processed")
    
    # Save files
    print("Save images ...")
    #sampled_images = images.sample(500)  
    images.to_pickle("preprocessed.pkl")
    print("Images saved")

