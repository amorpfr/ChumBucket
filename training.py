from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from itertools import combinations
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import numpy as np
import itertools
import pickle

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

def combi_lists(listX):
    """
    Program: Create lists of all possible featuresets
    Input: list of available features
    Output: list of all posiible featuresets
    """
    total_list = []
    for x in range(3,len(listX)+1):
        total_list = total_list + list(itertools.combinations(listX,x))
    total_list = tuple_to_list(total_list)
    return total_list

def test(images, used_features, clf):
    """
    Program: Computes the accuracy and logloss of a dataset consisisting images, features and classes
    Input: Dataframe consisting all data, list of names of features to be used
    Output: Accuracy
    """
    features = images[used_features]
    features = merge_features(features)
    labels = np.array(images['class'])
    trainX, testX, trainY, y_true = train_test_split(features, labels, test_size=0.2, random_state=42)
    scores = cross_validation.cross_val_score(clf, features, labels, cv=5, n_jobs=1);
    accuracy = np.mean(scores)
    return accuracy

def tuple_to_list(tuple_list):
    """
    Program: converts a list of tuples to a list of lists
    Input: List of tuples
    Output: Lists of lists
    """
    new_list = []
    for w in range(0,len(tuple_list)):
        new_list.append(list(tuple_list[w]))
    return new_list
 
def create_submission(images_test, used_features, model):
    """
    Program: Creates submission for kaggle
    Input: test images, features and model
    Output: submission dataframe and saves dataframe as csv
    """
    # features test set
    features = images_test[used_features]
    features = merge_features(features)
    
    # make submssion
    submission  = pd.DataFrame() 
    submission['images'] = images_test['image']
    submission['class'] = model.predict(features)
    submission.to_csv('submission.csv')   
    return submission     
    
def compare_accuracy(images, used_features, models, model_names):
    """
    Program: Compares the accuracy of different models and featuresets
    Input: Images, features, model and modelnames
    Output: Dataframe of the accuracy for each featureset and model
    """
    # loop through model
    result = {}
    for i in range(0, len(models)):
        model_dict = {}
        model = models[i]
        model_name = model_names[i]
        print("Running model : ", model_name, " ...")
        # loop through  features
        j=0
        for j in range(0, len(used_features)):
            print(j/float(len(used_features)))
            featureset = used_features[j]
            accuracy = test(images,featureset , model)
            model_dict[str(featureset)] = accuracy
            
        # save model results
        result[model_name] = model_dict
       
        print(model_name, " done!")
    # convert to dataframe
    result_df = pd.DataFrame.from_dict(result) 
    return result_df

def analyzing_models(images):
    """
    Program: Main program to analyze models and featuresets
    Input: Images
    Output: Accuracy dataframe of ceah molde and featureset
    """
    
    #  model list
    # canditates: LinearSVC(multi_class = "crammer_singer", random_state=0)
    # MLPClassifier(),  BernoulliNB(),   KNeighborsClassifier(n_neighbors=5),
    models = [RF(n_estimators=100, n_jobs=3), 
              #LogisticRegressionCV(multi_class="ovr", n_jobs=1, cv = 3),
              #LogisticRegressionCV(),
              ExtraTreesClassifier(n_estimators=100, n_jobs=3)
              ]
    
    model_names =['RandomForest','ExtraTrees']
    
    #  feature list
    available_features = ['haralick', 'zernike', 'binary pattern','ratio', 'pixels']
    combi_features = combi_lists(available_features)     
           
    # accruacy datafream of accuracys    
    accuracy_df = compare_accuracy(images, combi_features, models, model_names)
    return accuracy_df
    
def save_model(outputname, model):
    """
    Program: Dump the trained classifier with Pickle
    Input: path to save model, model
    Output: saved model as pickle
    """
    
    # Open the file to save as pkl file
    model_pkl = open(outputname, 'wb')
    pickle.dump(model, model_pkl)
    # Close the pickle instances
    model_pkl.close()
    
def train_model(images, used_features,outputpath):
    """
    Program: Trains a model and saves it as pickle
    Input: Images, path to save, model and used features
    Output: Model trained and model saved as pickle 
    """
    
    features_train = images[used_features]
    features_train = merge_features(features_train)
    labels = np.array(images['class'])
    clf = RF(n_estimators=100, n_jobs=3)
    model = clf.fit(features_train, labels) 
    save_model(outputpath, model)
    
if __name__ == "__main__":
    
    # load files
    print('Loading images ...')
    path = 'train100.pkl'
    images  = pd.read_pickle(path)
    print("Images loaded")
    
    # comparing models (comment code out when training)
    accuracy_df = analyzing_models(images)
         
    # create model(Run this when best model is determined)
    """
    used_features = ['haralick', 'zernike', 'binary pattern', 'ratio']
    outpath = 'modelHZBR.pkl'
    train_model(images, used_features, outpath)
    """
        
            
        