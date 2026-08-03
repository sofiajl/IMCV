###################################################
## BatchGeneratorTripletLoss
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
    def __init__(self, data, batch_size = 32, dim = (128, 128, 3), shuffle = True, num_files = None):
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
        self.NUM_FILES = 51 # Number of images by folder        
        

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
                - data_temp, string, file image
            - Output:
                - numpy array (batch_size, self.dim, channels), anchor image
                - numpy array (batch_size, self.dim, channels), positive image
                - numpy array (batch_size, self.dim, channels), negative image
        '''
        
        anchor = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.dim[2]))
        positive = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.dim[2]))
        negative = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.dim[2]))
        
        for i, (file_path, id) in enumerate(data_temp):

            # Choosing two random images
            pairs_person = random.sample(range(self.NUM_FILES), 2)

            # ANCHOR: Reading one image
            image_1 = imread(os.path.join(file_path, str(pairs_person[0]) + '.jpg'))
            X_image = np.array(image_1, dtype = np.uint8) / 255.0

            # POSITIVE: Reading another image
            image_2 = imread(os.path.join(file_path, str(pairs_person[1]) + '.jpg'))
            pos_image = np.array(image_2, dtype = np.uint8) / 255.0
            
            # NEGATIVE: Selecting image from different category
            neg_list = self.data.copy()
            del neg_list[i]
            
            neg_img_selected = random.choice(neg_list)[0]
            img_person = random.randint(0, self.NUM_FILES)
            neg_image = imread(os.path.join(neg_img_selected, str(img_person) + '.jpg')) / 255.0
            
            anchor[i, :, :, :], positive[i, :, :, :], negative[i, :, :, :] = X_image, pos_image, neg_image

        return anchor, positive, negative
    
    
    def __getitem__(self, index):
        '''
            Generates an index of batch data
            - Input:
                - index, int, batch index
            - Output:
                - numpy array (batch_size, self.dim, channels), anchor image
                - numpy array (batch_size, self.dim, channels), positive image
                - numpy array (batch_size, self.dim, channels), negative image
        '''
        
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        data_temp = [list(self.data)[k] for k in indexes]
        anchor, positive, negative = self.__data_generation(data_temp)
        
        return [anchor, positive, negative]
    
    
    