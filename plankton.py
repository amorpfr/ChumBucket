# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:47:29 2017

@author: daniel
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv

"""
def init(path):
    path = 'C:/Users/daniel/Documents/GitHub/ChumBucket/data/train_images/1.jpg'    
    rgb_img = plt.imread('path')
    plt.imshow(rgb_img)
    
    image_df = pd.read_pickle(path)
    user_id =  image_df['user_id'].astype(str)
    df_init = pd.DataFrame({'image_id': image_df['image_id'], 'user_id':user_id })
    
    return df_init
    """
path = 'C:/Users/daniel/Documents/GitHub/ChumBucket/data/train_images/2.jpg'    
imgFile = cv.imread(path)
cv.imshow('dsr_rt', imgFile)
cv.waitKey(0)
cv.destroyAllWindows()




import cv2
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images




rgb_img = plt.imread(path)
plt.imshow(rgb_img, cmap='gray')
img = np.array([rgb_img]) # Nu is het een package als het ware.

matrix = pd.DataFrame([rgb_img])
#index = pd.date_range(todays_date-datetime.timedelta(10), periods=10, freq='D')

#columns = ['A','B', 'C']
#df_ = pd.DataFrame(index=index, columns=columns)
df = pd.DataFrame([])
df.Columns.append([1])

# WE PROBEREN IETS ANDERS MET RESHAPE
h,w = rgb_img.shape
imgseries = np.reshape(rgb_img, [h*w])
pd.DataFrame(imgseries)
