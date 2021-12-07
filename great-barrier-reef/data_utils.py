# This is a class for interacting with the downloaded data
# Two variants are provided, one returning thumbnails for
# fine tuning the backbone and one returning full size images

from tensorflow.keras.utils import image_dataset_from_directory
import pandas as pd
import os
import glob
import json


# Classes in this file are called following
# data=data_utils.DataLoader*(input_file='tensorflow-great-barrier-reef')
# images = data.get_training(validation_split=0.2)


class DataLoader:
    def __init__(self, input_file='tensorflow-great-barrier-reef'):
        '''
        Instantiate a DataLoader pointed at the training data.
        Writing this class once to avoid replicating the same boilerplate in model building.
        Note that this class follows loads data lazily.

        Arguments:

        input_file : str
            Where the downloaded dataset from Kaggle lives locally.

        '''

        self.input_file = input_file
        self.batch_size = None
        self.validation_split = None
        self.seed = 150601497
        self.labels = pd.read_csv(os.path.join(self.input_file, 'train.csv'))


# This class will be a wrapper for getting the full-sized images into the ML models
# Intended to make things as atomic as possible for the high-level code


class DataLoaderFull(DataLoader):
    def __init__(self, **kwargs):
        '''
        Construct a DataLoaderThumbnail class with the same properties as DataLoader

        '''

        super().__init__(**kwargs)

        # TODO need to figure out how the files actually map to rows in the pandas database

        self.ifile = list(
            range(
                len(
                    glob.glob(os.path.join(self.input_file, 'train_images', '*/*.jpg'))
                ),
            )
        )

    def _load_dataset(self, batch_size=32, validation_split=0.2, shuffle=True):
        '''
        Internal method to load the dataset from the input files. Does nothing
        if the dataset has already been loaded with the same parameters.

        Arguments:

        batch_size : int
            Minibatch size to be used in SGD
        validation split : float
            Fraction of the data to keep as cross validation data.
        shuffle : bool
            Shuffle the data into minibatches, True by default.

        '''

        # Record what was done for posterity
        self.batch_size = batch_size
        self.validation_split = validation_split

        # Load the dataset
        self.training = image_dataset_from_directory(
            os.path.join(self.input_file, 'train_images'),
            labels=self.ifile,
            batch_size=batch_size,
            seed=self.seed,
            validation_split=self.validation_split,
            shuffle=shuffle,
            subset='training',
            image_size=(720, 1280),
        )

        self.validation = image_dataset_from_directory(
            os.path.join(self.input_file, 'train_images'),
            labels=self.ifile,
            batch_size=batch_size,
            seed=self.seed,
            validation_split=self.validation_split,
            shuffle=shuffle,
            subset='validation',
            image_size=(720, 1280),
        )

    def get_training(self, **kwargs):

        '''
        Fetch the training set. Keyword arguments are passed down to _load_dataset.

        '''

        if not hasattr(self, 'training'):
            self._load_dataset(**kwargs)

        return self.training

    def get_validation(self, **kwargs):

        '''
        Fetch the validation set. Keyword arguments are passed down to _load_dataset.

        '''

        if not hasattr(self, 'validation'):
            self._load_dataset(**kwargs)

        return self.validation

    def decode_label(self, label):
        '''
        Decode the label defined in the dataset creation into a list
        of bounding boxes.

        Assumes that the order of rows in the .npy file matches os.walk(),
        which empirically seems to be the case.

        Arguments:

        label: int
            Image number from the (sorted) dataset. This is decoded to a set of bounding boxes.

        '''

        st = self.labels['annotations'][label].replace('\'', '\"')
        return json.loads(st)


# This class returns a set of thumbnails that either do or do not have starfish.


class DataLoaderThumbnail(DataLoader):
    def __init__(self, **kwargs):
        '''
        Construct a DataLoaderThumbnail class with the same properties as DataLoaderFull

        '''

        super().__init__(**kwargs)

    def _load_dataset(self, batch_size=32, validation_split=0.2, shuffle=True):
        '''
        Internal method to create and load the thumbnail dataset from the input files.
        Skips thumbnail creation if those already exist on disk.

        Arguments:

        batch_size : int
            Minibatch size to be used in SGD
        validation split : float
            Fraction of the data to keep as cross validation data.
        shuffle : bool
            Shuffle the data into minibatches, True by default.

        '''

        if not os.path.exists(os.path.join(self.input_file, 'train_images_thumb')):

            # Now, work through the rows in the training labels and make cuts

            # TODO write this

            pass

        # Then load the thumnails

        # Load the dataset
        self.training = image_dataset_from_directory(
            os.path.join(self.input_file, 'train_images_thumb'),
            labels='inferred',
            label_mode='int',
            batch_size=batch_size,
            seed=self.seed,
            validation_split=0.2,
            shuffle=shuffle,
            subset='training',
        )

        self.validation = image_dataset_from_directory(
            os.path.join(self.input_file, 'train_images_thumb'),
            labels='inferred',
            label_mode='int',
            batch_size=batch_size,
            seed=self.seed,
            validation_split=0.2,
            shuffle=shuffle,
            subset='validation',
        )

    def get_training(self, **kwargs):

        '''
        Fetch the training set. Keyword arguments are passed down to _load_dataset.

        '''

        # Cut down to thumbnails
        if not hasattr(self, 'training'):
            self._load_dataset()

        return self.training

    def get_validation(self, **kwargs):

        '''
        Fetch the validation set. Keyword arguments are passed down to _load_dataset.

        '''

        # Cut down to thumbnails
        if not hasattr(self, 'validation'):
            self._load_dataset()

        return self.validation
