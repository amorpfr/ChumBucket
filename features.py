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
    image = resize(image, (maxPixel, maxPixel))
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
    pixels = image.flatten()
    return pixels

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
    
    
    #clf.fit(trainX, trainY)
    #y_pred = clf.predict(testX)
    #accuracy = float(accuracy_score(y_true, y_pred, normalize =True))
    #logloss = float(log_loss(y_true, y_pred))
    return accuracy

if __name__ == "__main__":
    # load files
    print('Loading images ...')
    path = 'images3000.pkl'
    images  = pd.read_pickle(path)
    print("Images loaded")
    
    # clean images
    print('Resizing images ...')
    maxPixel = 25
    images['resized'] = images['image_matrix'].apply(resize_image)
    print("Images resized")
    
    # extract features
    print('Extracting features ...')
    used_images = 'resized'
    #used_images = 'threshold'
    images['haralick'] = images['clean'].apply(haralick)
    images['ratio'] = images[used_images].apply(getMinorMajorRatio)
    images['image_size'] = images[used_images].apply(getSize)
    images['pixels'] = images[used_images].apply(pixel_feature)
    print("Features extracted")
   
    # test model
    print('Training model ...')
    features_to_use = ['haralick', 'ratio', 'image_size', 'pixels']
    accuracy = test(images, features_to_use)
    print("")
    print("Training done, testing accuracy: ", accuracy)
    print("") 
    
    