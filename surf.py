"""
This script extract the SURF features for all images
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.svm import LinearSVC
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier as RF
from skimage.transform import resize
from skimage import morphology
from skimage import measure

def resize_image(image):
    """
    Program: Resizes image based on global variable maxPixel
    Input: image
    Output: Resized(Squared) image
    """
    TERMINATOR = 100
    height = np.floor(TERMINATOR / (float(image.shape[0])/image.shape[1]))
    image = resize(image, (TERMINATOR, height), preserve_range= True, mode='wrap')
    return image

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


def pre_surf1(image):
    """
    Program: Preoprocess image suitable voor surf
    Input: Image
    Output: preprocessed image
    """   
    maxregion = get_important_region(image)
    
    small_img = image[maxregion.bbox[0]:maxregion.bbox[2],maxregion.bbox[1]:maxregion.bbox[3]]
    noNoise = np.where(small_img > np.mean(image),0.,1.0)
    plusje = np.where(small_img > np.mean(image),255.,0.)
    minimum = np.min(small_img)
    
    almost = (small_img * noNoise) + plusje - minimum
    pre_surf1 = resize_image(almost)

    return pre_surf1


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
    plt.imshow(labels)
    labels = np.uint8(labels)
    return labels

def init_surf(image_paths):
    """
    Program: Gettig surf descriptors from images
    Input: list of images
    Output: descriptors and descriptor lists
    """    
     # Create feature extraction and keypoint detector objects
    fea_det = cv2.FeatureDetector_create('FAST')
    des_ext = cv2.DescriptorExtractor_create("SURF")
    print(len(image_paths))
    print("Feature detection ...")
    # List where all the descriptors are stored
    des_list = []
    i=0
    for i in range(0,len(image_paths)):
        print(1,i)
        im= image_paths[i]
        im = pre_surf1(im)
        im = np.uint8(im)
        fea_det = cv2.SURF(10000)
        kpts = fea_det.detect(im)
        
        kpts, des = des_ext.compute(im, kpts)
        
        img =cv2.drawKeypoints(im,kpts)
        #plt.imshow(im)
        plt.imshow(img)
        
        
        if des is None:
            #print("HELLUEP")
            zero_des = np.zeros((1, 64), "float32")
            des_list.append((i, zero_des))   
        else:
            #print(np.shape(des))
            des_list.append((i, des))   
    print("Keypoints and descriptors extracted")   
    
    print("Stacking desriptors")
    # Stack all the descriptors vertically in a numpy array
    #print(des_list)
    descriptors = des_list[0][1]
    #print((descriptors))
    for image_path, descriptor in des_list[1:]:
        print(2,image_path)
        descriptors = np.vstack((descriptors, descriptor)) 
    descriptors = descriptors[~np.all(descriptors == 0, axis=1)]
    print("Descriptors extracted")
        
    return descriptors, des_list

def surf_method(image_paths,des_list, descriptors, k):
    """
    Program: Converting descriptors to feature vector per image
    Input: imagges, descriptors and aount of clusters
    Output: feature vectors of surf features
    """ 
    
    print("Clustering ...")
    # Perform k-means clustering
    #k = 12
    voc, variance = kmeans(descriptors, k, 1) 
    print("Clustering done ..")
    
    print("Making histogram ...")
    # Calculate the histogram of features
    im_features = np.zeros((len(image_paths), k), "float32")
    for i in range(len(image_paths)):
        #print(i)
        words, distacnce = vq(des_list[i][1],voc)
        for w in words:
            im_features[i][w] += 1
    print("Histogram done")    

    print("Making features ...")    
    """
    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')
    """
    # Scaling the words
    im_features1 = im_features
    stdSlr = StandardScaler().fit(im_features1)
    im_features1 = stdSlr.transform(im_features1)
    
    im_features1 = im_features1.tolist()
    im_features = im_features.tolist()
    
    print("Surf done")
    return im_features, im_features1, descriptors

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

def test(images, used_features):
    """
    Program: Computes the accuracy of a dataset consisisting images, features and classes
    Input: Dataframe consisting all data, list of names of features to be used
    Output: Accuracy
    """
    features = images[used_features]
    features = merge_features(features)
    labels = np.array(images['class'])
    
    clf = LinearSVC( random_state=0)
    #clf = ExtraTreesClassifier(n_estimators=100, n_jobs=3, verbose=True)
    #clf = RF(n_estimators=100, n_jobs=3)
    scores = cross_validation.cross_val_score(clf, features, labels, cv=4, n_jobs=1);
    accuracy = np.mean(scores)

    return accuracy

def compare_clusters(images, des_list, descriptors):
    """
    Program: Computes the accuracy of a dataset consisisting images, features and classes
    Input: Dataframe consisting all data, list of names of features to be used
    Output: Accuracy
    """
    nclusters = [125,140,150,160,180]
    accuracy_list =[]
    for i in range(0,len(nclusters)):
        k = nclusters[i]
        feats, feats_normalized, descriptors = surf_method(images['image_matrix'].tolist(),des_list, descriptors, k)
        #df = pd.DataFrame(descriptors)
        #df.to_pickle('descriptors2.pkl')
        images['normalized_sift'] = feats_normalized
        images['sift'] = feats
        
        used_features=['normalized_sift']
        accuracy = test(images, used_features)
        accuracy_list.append((k, accuracy))
        print(k, accuracy)

if __name__ == "__main__":
    # loading images
    images = pd.read_pickle('preprocessed.pkl')
    
    # getting descriptors
    descriptors,des_list = init_surf(images['image_matrix'].tolist())
    
    # extracting features
    feats, feats_normalized, descriptors = surf_method(images['image_matrix'].tolist(),des_list, descriptors, 140)
    
    # adding to dataframe
    images['normalized_surf'] = feats_normalized
    images['surf'] = feats
    
    # save dataframe
    images.to_pickle('surf.pkl')