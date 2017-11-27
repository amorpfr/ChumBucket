# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:51:07 2017

@author: AFR
"""
import pandas as pd
import pprint as pp
import numpy as np
import cv2

def to_image(image_name):
    path = 'C://Users//AFR//Documents//GitHub//ChumBucket//data//train_images//train_images//' + image_name
    img = cv2.imread(path,0)
    return img

if __name__ == "__main__":
    images  = pd.read_csv("C://Users//AFR//Documents//GitHub//ChumBucket//data//train_onelabel.csv", encoding='utf-8')
    images['image_matrix'] = images['image'].apply(to_image)

    #images.to_pickle("C://Users//AFR//Documents//GitHub//ChumBucket//data//images.pkl")
    