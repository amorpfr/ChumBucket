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
from sklearn.svm import LinearSVC
from sklearn.metrics import log_loss
from skimage.transform import resize
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.ensemble import RandomForestClassifier as RF
    
"""
#######################################################################
"""

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


"""
#######################################################################
"""

def haralick(image):
    """
    Program: calculate haralick texture features for 4 types of adjacency
    Input: Image
    Output: 13 haralick features (numpy array)
    """
    image = image.copy()
    image= image.astype(int)
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

def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss

def getSize(image):
    """
    Program: Get size of image
    Input: image
    Output: width and length in numpy array
    """
    (x,y) = np.shape(image)
    return np.array([x,y])

def resize_image(image):
    """
    Program: Resizes image based on global variable maxPixel
    Input: image
    Output: Resized(Squared) image
    """
    image = resize(image, (maxPixel, maxPixel), preserve_range= True)
    return image

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
    Radius is hetzelfde als straal. Aangezien de resized images 20*20 zijn,
    is er een straal nodig van 14.14 nodig om met die straal ook in de hoeken
    te komen.
    
    door cm = (10,10) toe te voegen veranderd er vrijwel niks
    
    Ook een border van 5pixels (255) rondom de images veranderd er niks. Ik
    dacht dat dit wel invloed zou hebben omdat de radius (tov center of mass)
    nu groter is dan de image lang en breed is.
    """
    
    zernike_features = mt.features.zernike_moments(image, radius=15, degree=8)
    return zernike_features

def linear_binary_pattern(image):
    lbp =   mt.features.lbp(image, radius=20, points=7, ignore_zeros=True)
    return lbp

def pftas(image):
    pft = mt.features.pftas(image)
    return pft
    
def get_dimension(image):
    """
    Program: Returns the largest dimension of the image
    Input: image
    Output: LArgest dimension 
    """
    shape = np.shape(image)
    return min(shape)


def test(images, used_features):
    """
    Program: Computes the accuracy and logloss of a dataset consisisting images, features and classes
    Input: Dataframe consisting all data, list of names of features to be used
    Output: Accuracy, kaggle log-loss
    """
    features = images[used_features]
    features = merge_features(features)
    labels = np.array(images['class'])
    trainX, testX, trainY, y_true = train_test_split(features, labels, test_size=0.2, random_state=42)
    #clf = LinearSVC(multi_class = "crammer_singer", random_state=0)
    #clf = SGDClassifier()
    clf = RF(n_estimators=100, n_jobs=3);
    scores = cross_validation.cross_val_score(clf, features, labels, cv=5, n_jobs=1);
    accuracy = np.mean(scores)
    #print(np.shape(features))
    #print(np.shape(labels))
    #trainset = features
    #trainset['class'] = labels
    #logloss = float(log_loss(y_true, y_pred))
    trainset=4
    return accuracy, trainset
    
if __name__ == "__main__":
    """
    TO DO:
        
    - functie Resize(skimage) parameters  - rgl 114
    - Voor haralick, zernike en binary pattern welke input meest geschikt
    - Sift/Surf features 
    - code om beste set of features te bepalen
    - code om beste beste classificatie te behalen bepalen
    
    """
    # load files
    print('Loading images ...')
    path = 'images1000.pkl'
    images  = pd.read_pickle(path)
    print("Images loaded")
    
    # clean images
    print('Resizing images ...')
    max_shapes = images['image_matrix'].apply(get_dimension)
    #maxPixel = int(np.mean(max_shapes)) #max-accuracy 0.36, min-accuracy 0.36
    #maxPixel = int(np.max(max_shapes)) #max-accuracy 0.35, min-accuracy 0.34
    #maxPixel = int(np.min(max_shapes)) #max-accuracy 0.36, min-accuracy 0.35
#    maxPixel = 20 # accuracy 0.38
#    images['resized'] = images['image_matrix'].apply(resize_image)
#    images['region'] = images['image_matrix'].apply(get_important_region)
    images['superb'] = images['image_matrix'].apply(superb)
    images['pre_haralick'] = images['image_matrix'].apply(pre_haralick)
    print("Images resized")
    
    # extract features
    print('Extracting features ...')
#    used_images = 'resized'
    #used_images = 'threshold'
#    images['ratio'] = images['image_matrix'].apply(getMinorMajorRatio)
#    images['pixels'] = images['resized'].apply(pixel_feature)
#    images['image_size'] = images['superb'].apply(getSize)
#    images['haralick'] = images['pre_haralick'].apply(haralick)
#    images['zernike'] = images['superb'].apply(zernike) #resized/threshold beste 
    images['binary_pattern'] = images['pre_haralick'].apply(linear_binary_pattern) #threshold beste
    #images['pftas'] = images['clean'].apply(pftas)
    
    print("Features extracted")
   
    # test model
    print('Training model ...')
    features_to_use = ['binary_pattern']
    accuracy, trainset = test(images, features_to_use)
    #trainset.to_pickle("train500.pkl")
    print("")
    print("Training done, testing accuracy: ", accuracy)
    print("") 
    
    """
    Heel veel informatie
    
    Resize:
        - Skimage resize werkt als volgt
        Interpolatie, geeft het terug als % (0 tot 1)
    
    Zernike moments:
        - https://www.pyimagesearch.com/2014/04/07/building-pokedex-python-indexing-sprites-using-shape-descriptors-step-3-6/
        - http://www.tandfonline.com/doi/pdf/10.1007/s11806-007-0060-x
        De grote zernike comparison
        - Only resized: 0.28
        - Only Superb: 0.25
        - Superb+resized: 0.32
        - resized maar binary: 0.27
        - Superb+ resized+ set radius to 15: 0.34
    
    """