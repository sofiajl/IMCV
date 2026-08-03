###################################################
## BatchGeneratorTripletSemiHardLoss
###################################################

## Libraries
import numpy as np
from skimage.io import imread 
import random
import tensorflow as tf
import os

## Class BatchGenerator
class BatchGenerator(tf.keras.utils.Sequence):
    '''
        Class for image neural net real time feeding
    '''
    def __init__(self, data, batch_size = 32, dim = (128, 128, 3), shuffle = True):
        '''
            Initializes BatchGenerator object
            - Input:
                - data, list of tuples, file names and ids
                - batch_size, int, number of samples per batch
                    Default 32
                - dim, list, dimension of the images
                    Default (128, 128, 3)
                - shuffle, boolean, shuffle data after each epoch
                    Default True
        '''
        
        self.data = data
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        

    def on_epoch_end(self):
        '''
            Updates indexes after each epoch
        '''
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
            
    def __len__(self):
        '''
            Denotes the number of batches per epoch
            - Output:
                - int, number of batches
        '''
        return int(np.floor(len(self.data) / self.batch_size))


    
    def __data_generation(self, data_temp):
        '''
            Generates data batch
            - Input:
                - data_temp, list of tuples, file path and id person
            - Output:
                - numpy array (batch_size, self.dim), image
                - numpy array (batch_size), person id
        '''
        
        X = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.dim[2]))
        y = np.zeros((self.batch_size))
        
        i = 0
        for file_path, id in data_temp:
            
            pairs_person = random.sample(range(51), 2)
            
            image_1 = imread(os.path.join(file_path, str(pairs_person[0]) + '.jpg'))
            X_image_1 = np.array(image_1, dtype = np.uint8) / 255.0
        
            image_2 = imread(os.path.join(file_path, str(pairs_person[1]) + '.jpg'))
            X_image_2 = np.array(image_2, dtype = np.uint8) / 255.0
            
            X[i, :, :, :], y[i] = X_image_1, id
            X[i + 1, :, :, :], y[i + 1] = X_image_2, id
            
            i += 2

        return X, y
    
    
    def __getitem__(self, index):
        '''
            Generates an index of batch data
            - Input:
                - index, int, batch index
            - Output:
                - numpy array (batch_size, self.dim), image
                - numpy array (batch_size), person id
        '''
        
        indexes = self.indexes[index * int(self.batch_size / 2): (index + 1) * int(self.batch_size / 2)]
        data_temp = [list(self.data)[k] for k in indexes]
        X, y = self.__data_generation(data_temp)
        
        return [X, y]
    
    
    
    