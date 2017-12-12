"""
This file can be used to visualize the precision and recall of a model.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier as RF


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
        feature_matrix[i] = merge(featureset.iloc[i].tolist())
    return feature_matrix    


def test(images, used_features):
    """
    Program: Predicts the plankton class based on the features retrieved from each image
    Input: Dataframe consisting all data, list of names of features to be used
    Output: Predicted plankton class per image
    """
    features = images[used_features]
    features = merge_features(features)
    labels = np.array(images['class'])
    
    clf = RF(n_estimators=100, n_jobs=3);
    predicted = cross_val_predict(clf, features, labels, cv=5, n_jobs= 1)
    
    return predicted


def Precision_Recall(images):

    y_count = images.groupby(['predicted']).count()
    x_count = images.groupby(['class']).count()
    y = y_count.iloc[:,1]; x = x_count.iloc[:,1]
    goed = pd.concat([x,y], axis=1)
    df = pd.DataFrame(goed)
    df.columns = ['x', 'y']
    df = df.fillna(0.)
    df['error'] = abs(df.x-df.y)
    
    total_goed = pd.DataFrame(range(0,121))
    for i in range(0,121):

        fout_pred = images.loc[images['class'] == i].predicted
        if np.sum(fout_pred.value_counts().index == i) == 0:
            total_goed.loc[i] = 0
        else:
            total_goed.loc[i] = fout_pred.value_counts().loc[i]
        
    df['correct'] = total_goed
    df['recall'] = df.correct/df.x
    df['precision'] = df.correct/df.y
    df['scaled'] = np.log(df.x)
    df = df.fillna(0.)
    df = df.sort_values(by = 'x', ascending= False)
    
    return df

def scatterplot(df):
    plt.scatter(df.scaled, df.recall, marker="x", color="red", s=30, linewidths=1, label="Recall")
    plt.scatter(df.scaled, df.precision, marker="+", color="blue", s=30, linewidths=1, label="Precision")

    plt.legend(scatterpoints=1)
    rec_fit = np.polyfit(df.scaled, df.recall, deg=1)
    precis_fit = np.polyfit(df.scaled, df.precision, deg=1)
    
    plt.plot(df.scaled, rec_fit[0] * df.scaled + rec_fit[1], color='red')
    plt.plot(df.scaled, precis_fit[0] * df.scaled + precis_fit[1], color='blue')
    
    plt.xlabel("Common logarithm of class size")
    plt.ylabel("Precision & recall fractions")
    plt.tight_layout()

    plt.show()
    
if __name__ == "__main__":
    
    # Evaluate computation time
    start_time = time.time()

    # load files
    print('Loading images ...')
    path = 'images_features_final.pkl'
    images  = pd.read_pickle(path)
    print("Images loaded")
    
    # test model
    print('Training model ...')
    features_to_use = ['ratio','image_size','haralick','zernike','binary_pattern', 'binary_pattern_small']
    images['predicted'] = test(images, features_to_use)
    print("Training done")

    # Create dataframe and calculate precision & recall per class    
    print('create dataframe ...')
    df = Precision_Recall(images)
    print('created dataframe')
    
    # Visualize precision & recall
    print('plot figure')
    scatterplot(df)    
    
    print("--- %s seconds ---" % (time.time() - start_time))
