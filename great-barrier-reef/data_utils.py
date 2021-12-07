#
# This is a class for interacting with the downloaded data
#
# For now it is a wrapper around the tf.keras utility, in the future it will be tweaked to return thumbnails.
#

from tensorflow.keras.utils import image_dataset_from_directory
import pandas as pd
import os
import glob
import numpy as np

# This class will be a wrapper for getting the full-sized images into the ML models
# Intended to make things as atomic as possible for the high-level code
#
# data=data_utils.DataLoader(input_file='tensorflow-great-barrier-reef')
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
        self.ground_truth = None
        self.seed = 150601497

    def _load_dataset(
        self, batch_size=32, validation_split=0.2, shuffle=True, reseed=None
    ):
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

        # Make sure the params haven't changed.
        if self.batch_size == batch_size and self.validation_split == validation_split:
            return

        # Reseed the data if requested
        if reseed is not None:
            self.seed = reseed

        # Record what was done for posterity
        self.batch_size = batch_size
        self.validation_split = validation_split

        # Load the labels and count files
        self.labels = pd.read_csv(os.path.join(self.input_file, 'train.csv'))
        self.ifile = list(
            range(
                len(
                    glob.glob(os.path.join(self.input_file, 'train_images', '*/*.jpg'))
                ),
            )
        )

        # Load the dataset
        self.training = image_dataset_from_directory(
            os.path.join(self.input_file, 'train_images'),
            labels=self.ifile,
            batch_size=batch_size,
            seed=self.seed,
            validation_split=0.2,
            shuffle=shuffle,
            subset='training',
        )

        self.validation = image_dataset_from_directory(
            os.path.join(self.input_file, 'train_images'),
            labels=self.ifile,
            batch_size=batch_size,
            seed=self.seed,
            validation_split=0.2,
            shuffle=shuffle,
            subset='validation',
        )

    def decode_label(label):
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


# This class will return a set of thumbnails that either do or do not have starfish.
# I haven't figured out if we actually need this to pre-train any part of the Faster R-CNN
# so I haven't written it yet. It will have the same API is DataLoader.


class DataLoaderThumbnail(DataLoader):
    def __init__(self, **kwargs):
        '''
        Construct a DataLoaderThumbnail class with the same properties as DataLoader

        '''

        super().__init__(**kwargs)

    def _create_thumbnails(self, **kwargs):
        '''
        Applies the mapping in transform_full_to_thumbnail() to the elements
        of the original dataset.

        Keyword arguments are passed down to that function.

        '''

        self.training_thumbs = self.training.map(transform_full_to_thumbnail)

        self.validation_thumbs = self.validation.map(transform_full_to_thumbnail)

    def get_training(self, **kwargs):

        '''
        Fetch the training set. Keyword arguments are passed down to _load_dataset.

        Applies a map() operation on the existing dataset to turn it into thumbnails.

        '''

        # Load the dataset in the same way as the DataLoader
        if not hasattr(self, 'training'):
            super()._load_dataset(**kwargs)

        # Cut down to thumbnails
        if not hasattr(self, 'training_thumbs'):
            self._create_thumbnails()

        return self.training_thumbs

    def get_validation(self, **kwargs):

        '''
        Fetch the validation set. Keyword arguments are passed down to _load_dataset.

        Applies the map() operation on the existing dataset to turn it into thumbnails.

        '''

        if not hasattr(self, 'validation'):
            super()._load_dataset(**kwargs)

        # Cut down to thumbnails
        if not hasattr(self, 'validation_thumbs'):
            self._create_thumbnails()

        return self.validation_thumbs


# Method for taking an image and label, and converting to a thumbnail
def transform_full_to_thumbnail(image, label):

    return image, label
