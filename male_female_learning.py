import pandas as pd
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import os
import cv2
import csv

%matplotlib inline
#creating a Dataframe for refering the gender data
df=pd.read_csv('Age_gender_info.csv')
df.head()

#visualization of dataframe
sns.catplot(data=df,x='Gender',kind='count')


#creating a dataframe with reference to file path and gender information
datadir = put_filepath #eg(r"c:/documents/")
categories=[]
filenames=[]
df.ID=df.ID.astype(str)
for value in df.index:
    img_call=df.loc[value]['ID']
    if len(img_call)==8:
        filenames.append(os.path.join(datadir,'GEI',img_call + '.png'))
    elif len(img_call)==7:
        filenames.append(os.path.join(datadir,'GEI','0' + img_call + '.png'))
    elif len(img_call)==6:
        filenames.append(os.path.join(datadir,'GEI','00' + img_call + '.png'))
    elif len(img_call)==5:
        filenames.append(os.path.join(datadir,'GEI','000' + img_call + '.png'))
    elif len(img_call)==4:
        filenames.append(os.path.join(datadir,'GEI','0000' + img_call + '.png'))
    elif len(img_call)==3:
        filenames.append(os.path.join(datadir,'GEI','00000' + img_call + '.png'))
    elif len(img_call)==2:
        filenames.append(os.path.join(datadir,'GEI','000000' + img_call + '.png'))
    else:
        filenames.append(os.path.join(datadir,'GEI','0000000' + img_call + '.png'))
    category=df.loc[value]['Gender']
    if category == 'M':
        categories.append(1)
    else:
        categories.append(0)
df_data = pd.DataFrame({'filename': filenames,'category': categories})
df_data['category'].value_counts().plot.bar()


#Function to read image>>resize image>>saving image data to a array
def create_image_data(filename,category):
    image_data=[]
    class_num=category
    for img in filename:
        img_array=cv2.imread(img,cv2.IMREAD_GRAYSCALE)
        IMG_SIZE=88
        new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
        image_data.append([new_array,class_num])
    return image_data


#creating a training data array with image and gender info.
training_data=create_image_data(df_train['filename'],df_train['category'])
len(training_data)


#creating a testing data array with image and gender info.
testing_data=create_image_data(df_test['filename'],df_test['category'])
len(testing_data)

#creating a numpy zip file for training and testing data
np.save('training_data.npy', training_data)
np.save('testing_data.npy',testing_data)