###################################################
## BatchGeneratorArcFace
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
    
    def __init__(self, file_images: np.ndarray, batch_size: int = 32, shuffle: bool = True):
        '''
        Initializes BatchGeneratorCactus object
        - Input:
            - file_images, list, list with the file names
            - batch_size, int, number of samples per batch
            Default 32
            - shuffle, boolean, shuffle data after each epoch
            Default True
        '''
        
        self.file_images = file_images
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.file_images))
        
        def __len__(self):
            '''
            Denotes the number of batches per epoch
            - Output:
                - int, number of batches
            '''
            
            return int(np.ceil(len(self.indexes) / self.batch_size))
            
        def on_epoch_end(self):
            '''
            Updates indexes after each epoch
            '''
            
            if self.shuffle:
                np.random.shuffle(self.indexes)
                
        def __getitem__(self, idx: int):
            '''
            Generates data batch
            - Input:
                - idx, int, batch index
            - Output:
                - numpy array (batch_size, self.dim), image
                - numpy array (batch_size), one hot encoder of the person id
            '''
            
            indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
            file_images = [self.file_images[k] for k in indexes]
            X = []
            y = np.zeros((len(file_images), MAX_SUBJECTS), dtype=np.float32)
            for i, file_image in enumerate(file_images):
                X_i = imread(file_image).astype("float32")
                X += [X_i]
                y[i, int(file_image.split("\\")[1].split(".")[0])] = 1.
                
            X = np.stack(X)
            
            return (X / 255., y), y






