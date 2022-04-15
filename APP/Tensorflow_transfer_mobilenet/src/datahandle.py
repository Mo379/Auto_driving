import os
import sys
import glob
import numpy as np
from numpy import genfromtxt
from numpy import asarray
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class DataLoader:
    def __init__(self, directory, images_dir='',labels_file=''):
        self.directory = directory
        self.images_dir = images_dir
        self.labels_file= labels_file

    def Load_all_batches(self, data, data_shape=(-1,1)):
        loaded_set_features = []
        loaded_set_labels= []
        for batch in data:
            X = []
            Y = []
            for instance in batch:
                print(instance)
                link = instance[0]
                image = Image.open(link)
                data = asarray(image)
                if data.shape[2] == 4:
                    data = data[:,:,:-1]
                if data_shape != 'original':
                    data = data.reshape(data_shape)
                X.append(data)
                image_number = instance[1]
                angle = instance[2]
                speed = instance[3]
                Labels = np.array([angle,speed], dtype=np.float32)
                Y.append(Labels)
            loaded_set_features.append(X)
            loaded_set_labels.append(Y)
        loaded_set_features = np.array(loaded_set_features,dtype=np.float32)
        loaded_set_labels= np.array(loaded_set_labels,dtype=np.float32)
        return loaded_set_features,loaded_set_labels

    def Load_batch(self, batch, data_shape=(-1,1), augmentation=False, augmentation_pass = 3):
        X = []
        Y = []
        for instance in batch:
            link = instance[0]
            image = Image.open(link)
            data = asarray(image)
            if data.shape[2] == 4:
                data = data[:,:,:-1]
            if data_shape != 'original':
                data = data.reshape(data_shape)
            X.append(data)
            image_number = instance[1]
            angle = instance[2]
            speed = instance[3]
            Labels = np.array([angle,speed], dtype=np.float32)
            Y.append(Labels)
        if augmentation:
            X,Y = self._augment_batch(X,Y, n=augmentation_pass)
        loaded_set_features = np.array(X,dtype=np.float32)
        loaded_set_labels= np.array(Y,dtype=np.float32)
        return loaded_set_features,loaded_set_labels
    def _augment_batch(self, X,Y,n):
        rotation = layers.experimental.preprocessing.RandomRotation(0.15)
        zoom = layers.experimental.preprocessing.RandomZoom(0.15)
        contrast = layers.RandomContrast(0.15)
        data = X
        X = X * n
        x = rotation(X)
        x = zoom(x)
        x = contrast(x)
        x = x.numpy().tolist()
        augmented_X= data + x 
        augmented_Y= Y*(n+1)
        return augmented_X, augmented_Y

    def Load_batch_quiz(self, batch, data_shape=(-1,1)):
        X = []
        Image_order = []
        for instance in batch:
            link = instance[0]
            image = Image.open(link)
            data = asarray(image)
            if data.shape[2] == 4:
                data = data[:,:,:-1]
            if data_shape != 'original':
                data = data.reshape(data_shape)
            image_number = instance[1]
            X.append(data)
            Image_order.append(image_number)
        X = np.array(X)
        Image_order = np.array(Image_order)
        return X,Image_order


    def LoadCollectedData_info(self,split,batch_size):
        self._get_collected_imgsinfo_train()
        if split:
            self._train_test_split(split=split)
        if batch_size:
            self._batch_imgsinfo_train(batch_size=batch_size)
        return self.train_images_information, np.array([self.test_images_information])


    def LoadModelData_info(self, split, batch_size):
        self._get_imgsinfo_train()
        if split:
            self._train_test_split(split=split)
        if batch_size:
            self._batch_imgsinfo_train(batch_size=batch_size)
        return self.train_images_information, np.array([self.test_images_information])

    def LoadQuizData_info(self):
        self._get_imgsinfo_test()
        return self.quiz_images_information

    def _train_test_split(self, split):
        n = len(self.train_images_information)
        split_index = int(np.ceil(n * split))
        train_split = self.train_images_information[0:split_index]
        test_split = self.train_images_information[split_index+1:n]
        self.train_images_information = train_split
        self.test_images_information = test_split

    def _batch_imgsinfo_train(self, batch_size):
        n = len(self.train_images_information)
        if n % batch_size ==0:
            batched_imgsinfo_train = np.reshape(self.train_images_information,
                    (int(n/batch_size),-1,4)
                    )
        else: 
            while n % batch_size != 0:
                n -= 1
            batched_imgsinfo_train = np.reshape(
                    self.train_images_information[0:n],
                    (int(n/batch_size),-1,4)
                    )
        self.train_images_information = batched_imgsinfo_train

    def _get_collected_imgsinfo_train(self):
        #Get the absolute paths of the data
        absolute_paths = glob.glob(self.directory+"/*.png") 
        #getting the labels
        self.train_images_information= []
        for path in absolute_paths:
            parts = path.split('/')[-1].split('.')[0].split('_')
            image_number= parts[0]
            angle = parts[1]
            angle = round((int(angle)-50)/(120-50),3)
            speed = parts[2]
            speed = round(int(speed)/35, 3)
            if speed < 0.5:
                speed = round(np.random.uniform(low=0.0, high=0.15),3)
            else:
                speed = round(np.random.uniform(low=0.85, high=1),3)
            information =[ 
                    path,
                    image_number,
                    angle,
                    speed
                    ]
            self.train_images_information.append(information)

    def _get_imgsinfo_train(self):
        #Get the absolute paths of the data
        absolute_paths = glob.glob(self.directory+self.images_dir+"/*.png") 
        #getting the labels
        labels = self._get_labels()
        df = pd.DataFrame(labels, columns = ['index','angle','speed'])
        df = df.set_index('index')
        self.train_images_information= []
        for path in absolute_paths:
            parts= path.split('/')
            image_name = parts[-1]
            image_number = int(image_name.split('.')[0])
            if image_number in df.index:
                image_label = df.loc[image_number]
            else:
                continue
            angle = image_label[0]
            speed = image_label[1]
            if speed < 0.5:
                speed = round(np.random.uniform(low=0.0, high=0.15),3)
            else:
                speed = round(np.random.uniform(low=0.85, high=1),3)

            information =[ 
                    path,
                    image_number,
                    angle,
                    speed
                    ]
            self.train_images_information.append(information)

    def _get_imgsinfo_test(self):
        #Get the absolute paths of the data
        absolute_paths = glob.glob(self.directory+self.images_dir+"/*.png") 
        #getting the labels
        self.quiz_images_information= []
        for path in absolute_paths:
            parts= path.split('/')
            image_name = parts[-1]
            image_number = int(image_name.split('.')[0])
            information =[ 
                    path,
                    image_number,
                    ]
            self.quiz_images_information.append(information)

    def _get_labels(self):
        labels = genfromtxt(self.directory+self.labels_file , delimiter=',' , skip_header=1, dtype=np.float32) 
        return labels


