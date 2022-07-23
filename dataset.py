import numpy as np
import keras
import cv2
from keras.utils import data_utils
import tensorflow as tf

class DataGenerator(data_utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x_idx, Y, X_dir, batch_size=32, input_dim=(320,320,3), out_dim=(20,20,8), scale_by=256, shuffle=True):
        'Initialization'
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.batch_size = batch_size

        self.Y = Y
        self.X_dir = X_dir

        self.list_IDs = x_idx
        self.scale_by = scale_by
        
        self.shuffle = shuffle
        self.on_epoch_end()

        print(self.Y.shape, len(self.indexes), len(self.list_IDs))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.input_dim))
        y = np.empty((self.batch_size, *self.out_dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i] = np.load(self.X_dir + str(self.list_IDs[ID]) + '.npy')/self.scale_by

            # Store class
            y[i] = self.Y[ID]

        return X, y



class XGen(data_utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, X_dir, y, scale):
        'Initialization'
        self.list_IDs = list_IDs
        self.X_dir = X_dir
        X = np.load(self.X_dir + str(0) + '.npy')
        self.ndim = len(X.shape)+1
        self.shape = (len(list_IDs), *X.shape)
        self.dim = X.shape
        self.scale = scale
        self.y = y
        self.dimY = y.shape
        print('in shape', self.shape)
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs)

    def __getitem__(self, indexes):
        'Generate one batch of data'
        # Generate indexes of the batch
        if isinstance(indexes, int):
            #X = np.empty((*self.dim))
            X = np.load(self.X_dir + str(self.list_IDs[indexes]) + '.npy')
            Y = self.y[indexes]
        else:
            X = np.empty((len(indexes), *self.dim))
            Y = self.y[indexes]
            for i, index in enumerate(indexes):
                X[i] = np.load(self.X_dir + str(self.list_IDs[index]) + '.npy')/self.scale
        print(X.shape, Y.shape)
        X = tf.cast(X, tf.float32)
        return (X, Y)


class XGenTraffic(data_utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, X_dir, scale):
        'Initialization'
        self.list_IDs = list_IDs
        self.X_dir = X_dir
        X = cv2.imread(self.X_dir + str(0) + '.jpeg')
        self.ndim = len(X.shape)+1
        self.shape = (len(list_IDs), *X.shape)
        self.dim = X.shape
        self.scale = scale

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs)

    def __getitem__(self, indexes):
        'Generate one batch of data'
        # Generate indexes of the batch
        if isinstance(indexes, int):
            #X = np.empty((*self.dim))
            #X = np.load(self.X_dir + str(self.list_IDs[indexes]) + '.npy')
            X = cv2.imread(self.X_dir + str(self.list_IDs[indexes]) + '.jpeg')
        else:
            X = np.empty((len(indexes), *self.dim))
            for i, index in enumerate(indexes):
                X[i] = np.array(cv2.imread(self.X_dir + str(self.list_IDs[indexes]) + '.jpeg')/self.scale, dtype=np.float)
        return X