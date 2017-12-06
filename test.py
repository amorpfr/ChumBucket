import pandas as pd
import numpy as np
import pickle
def merge(row):
    """
    Program: Merge multiple numpy arrays in one numpy array
    Input: List of numpy arrays
    Output: One numpy array of the merged arrays
    """
    merged = np.concatenate(row)
    return merged

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
    submission.to_csv('submission.csv', index = False)   
    return submission    



if __name__ == "__main__":
    
    # load model
    print('Loading model ...')
    path = 'modelHZBR.pkl'
    model_pkl = open(path, 'rb')
    model = pickle.load(model_pkl)
    print("Model loaded")
     
    # load test images
    print("Loading test images ...")
    path_test1 = 'test1.pkl'
    path_test2 = 'test2.pkl'
    images_test1 = pd.read_pickle(path_test1)
    images_test2 = pd.read_pickle(path_test2)
    images_test = images_test1.append(images_test2, ignore_index=True)
    print("Loading test images")
    
    # create submission
    print("Making submission ...")
    used_features = ['haralick', 'zernike', 'binary pattern', 'ratio']
    submission = create_submission(images_test, used_features, model)  
    print("Submission done")
    