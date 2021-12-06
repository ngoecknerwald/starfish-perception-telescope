#
# This is a class for interacting with the downloaded data
#
# For now it is a wrapper around the tf.keras utility, in the future it will be tweaked to return thumbnails.
#

from tensorflow.keras.utils import image_dataset_from_directory

# This class will be a wrapper for getting the full-sized images into the ML models
# Intended to make things as atomic as possible for the high-level code
#
# data=data_utils.DataLoader(input_file='tensorflow-great-barrier-reef')
# images = data.get_training(validation_split=0.2)

class DataLoader:
    def __init__(self, input_file='tensorflow-great-barrier-reef'):
        '''
        Instantiate a DataLoader pointed at the training data.
    
        Writing this once to avoid replicating the same boilerplate in model building.
        
        Arguments:
        
            input_file : str
                Where the downloaded dataset from Kaggle lives locally.

        '''
        self.input_file = input_file

    def _load_dataset(self, batch_size=32, video_number=-1, validation_split=0.2):

        if video_number >= 0:
            infile = os.path.join(
                self.input_file, 'train_images', 'video_%d' % video_number
            )
        else:
            infile = self.input_file

        self.training = image_dataset_from_directory(
            infile,
            labels=None,
            batch_size=batch_size,
            validation_split=0.2,
            shuffle=False,
            subset='training',
        )
        self.validation = image_dataset_from_directory(
            infile,
            labels=None,
            batch_size=batch_size,
            validation_split=0.2,
            shuffle=False,
            subset='validation',
        )

    def _load_labels(self, batch_size=32, video_number=-1, validation_split=0.2)
        
        pass
        
    def get_training(self, **kwargs):

        '''
        Fetch the training set. Keyword arguments are passed down to 
        
        '''
        
        
        if not hasattr(self, 'train_dataset'):
            self._load_dataset(**kwargs)

        return self.training

    def get_validation(self, **kwargs):

        if not hasattr(self, 'validation'):
            self._load_dataset(**kwargs)

        return self.validation

# This class will return a set of thumbnails that either do or do not have starfish.
# I haven't figured out if we actually need this to pre-train any part of the Faster R-CNN
# so I haven't written it yet. It will have the same API is DataLoader

class DataLoaderThumbnail:
    
    def __init__:
        pass
