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
    path = './data/train_images/' + image_name
    img = cv2.imread(path,0)
    return img

if __name__ == "__main__":
    images  = pd.read_csv("./data/train_onelabel.csv", encoding='utf-8')
    images['image_matrix'] = images['image'].apply(to_image)
    end_df = images[:100]   
    end_df.to_pickle("images_test.pkl")
    