# This is a class for interacting with the downloaded data
# Two variants are provided, one returning thumbnails for
# fine tuning the backbone and one returning full size images

import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
import pandas as pd
import os
import glob
import json
import numpy as np
from matplotlib import image
from PIL import Image  # tooooo many things called "image"

# Classes in this file are called following
# data=data_utils.DataLoader*(input_file='tensorflow-great-barrier-reef')
# images, labels = data.get_training(validation_split=0.2)


class DataLoader:
    def __init__(self, input_file="tensorflow-great-barrier-reef"):
        """
        Instantiate a DataLoader pointed at the training data.
        Writing this class once to avoid replicating the same boilerplate in model building.
        Note that this class follows loads data lazily.

        Arguments:

        input_file : str
            Where the downloaded dataset from Kaggle lives locally.

        """

        self.input_file = input_file
        self.batch_size = None
        self.validation_split = None
        self.seed = 150601497

        self.labels = pd.read_csv(os.path.join(self.input_file, "train.csv"))

        # Make the labels a tensor to be used in compiled models
        # Indices are image #, annotation #, xywh
        # Note that ragged tensors blow things up in the computation
        # graph later, for unclear reasons
        self.label_tensor = tf.constant(
            [
                self.label_to_tensor(annotation)
                for annotation in self.labels["annotations"]
            ],
            dtype="float32",
        )

    @staticmethod
    def label_to_tensor(label, max_annotations=32):
        """
        Convert the annotation string format to a nested list
        to be used in the annotation constant tensor.

        """
        annotations = [
            [a["x"], a["y"], a["width"], a["height"]]
            for a in json.loads(label.replace("'", '"'))
        ]

        return (
            annotations
            + [
                [0, 0, 0, 0],
            ]
            * (max_annotations - len(annotations))
        )


class DataLoaderFull(DataLoader):
    def __init__(self, **kwargs):
        """
        Construct a DataLoaderThumbnail class with the same properties as DataLoader.

        Creates an index for matching elements in the dataset with rows in the label database.

        """

        super().__init__(**kwargs)

        # Indices of filenames that then get handed into the dataset loader
        filenames = [
            "video_%d/%d.jpg"
            % (self.labels["video_id"][index], self.labels["video_frame"][index])
            for index in range(len(self.labels))
        ]
        self.ifile = list(np.argsort(filenames))

    def _load_dataset(self, batch_size=4, validation_split=0.2, shuffle=True):
        """
        Internal method to load the dataset from the input files. Does nothing
        if the dataset has already been loaded with the same parameters.

        Arguments:

        batch_size : int
            Minibatch size to be used in SGD
        validation split : float
            Fraction of the data to keep as cross validation data.
        shuffle : bool
            Shuffle the data into minibatches, True by default.

        """

        # Record what was done for posterity
        self.batch_size = batch_size
        self.validation_split = validation_split

        # Load the dataset
        self.training = image_dataset_from_directory(
            os.path.join(self.input_file, "train_images"),
            labels=[i for i in self.ifile],
            batch_size=batch_size,
            seed=self.seed,
            validation_split=self.validation_split,
            shuffle=shuffle,
            subset="training",
            image_size=(720, 1280),
        )

        self.validation = image_dataset_from_directory(
            os.path.join(self.input_file, "train_images"),
            labels=[i for i in self.ifile],
            batch_size=batch_size,
            seed=self.seed,
            validation_split=self.validation_split,
            shuffle=shuffle,
            subset="validation",
            image_size=(720, 1280),
        )

    def get_training(self, **kwargs):

        """
        Fetch the training set. Keyword arguments are passed down to _load_dataset.

        """

        if not hasattr(self, "training"):
            self._load_dataset(**kwargs)

        return self.training

    def get_validation(self, **kwargs):

        """
        Fetch the validation set. Keyword arguments are passed down to _load_dataset.

        """

        if not hasattr(self, "validation"):
            self._load_dataset(**kwargs)

        return self.validation

    @tf.function
    def decode_label(self, label):
        """
        Decode the label defined in the dataset creation into a list
        of bounding boxes.

        Arguments:

        label: tf.tensor(int32)
          Index tensor of size (batch, ) from the (sorted) dataset. This is decoded to a set of bounding boxes.

        """

        print("Python interpreter in DataLoader.decode_label()")

        return tf.gather(self.label_tensor, label)


# This class returns a set of thumbnails that either do or do not have starfish.
class DataLoaderThumbnail(DataLoader):
    def __init__(self, **kwargs):
        """
        Construct a DataLoaderThumbnail class with the same properties as DataLoaderFull

        """

        super().__init__(**kwargs)

    def _load_dataset(
        self, batch_size=64, validation_split=0.2, shuffle=True, image_size=(96, 96)
    ):
        """
        Internal method to create and load the thumbnail dataset from the input files.
        Skips thumbnail creation if those already exist on disk.

        Arguments:

        batch_size : int
            Minibatch size to be used in SGD
        validation split : float
            Fraction of the data to keep as cross validation data.
        shuffle : bool
            Shuffle the data into minibatches, True by default.
        image_size : tuple
            Spatial size of the images to cut out. (128, 128) is a good default for this dataset.

        """

        if not os.path.exists(os.path.join(self.input_file, "train_images_thumb")):

            os.makedirs(
                os.path.join(self.input_file, "train_images_thumb", "background")
            )
            os.makedirs(os.path.join(self.input_file, "train_images_thumb", "starfish"))

            # Now, work through the rows in the training labels and make cuts
            for index in range(len(self.labels)):

                # Monitor progress
                if index % 1000 == 0:
                    print(index)

                # Grab the annotations and the file name
                annotation = json.loads(
                    self.labels["annotations"][index].replace("'", '"')
                )
                filename = os.path.join(
                    self.input_file,
                    "train_images",
                    "video_%d/%d.jpg"
                    % (
                        self.labels["video_id"][index],
                        self.labels["video_frame"][index],
                    ),
                )

                # Load the image
                local_image = image.imread(filename)

                # Approximately half of the training data contains starfish,
                if (
                    len(annotation) > 0
                ):  # If there are starfish, then pick one at random
                    choice = np.random.randint(len(annotation))
                    xmin = annotation[choice]["x"]
                    ymin = annotation[choice]["y"]
                    local_size = (
                        annotation[choice]["width"],
                        annotation[choice]["height"],
                    )

                    outdir = os.path.join(
                        self.input_file, "train_images_thumb", "starfish"
                    )
                else:  # Use this frame as a background sample and randomly crop
                    xmin = np.random.randint(local_image.shape[1] - image_size[1])
                    ymin = np.random.randint(local_image.shape[0] - image_size[0])
                    outdir = os.path.join(
                        self.input_file, "train_images_thumb", "background"
                    )
                    local_size = image_size

                # Crop and save the results
                slice_image = Image.fromarray(
                    local_image[
                        ymin : ymin + local_size[1], xmin : xmin + local_size[0], :
                    ]
                )
                slice_image.save(os.path.join(outdir, "%d.jpg" % index))

        # Then load the thumnails
        self.training = image_dataset_from_directory(
            os.path.join(self.input_file, "train_images_thumb"),
            labels="inferred",
            label_mode="binary",
            class_names=("background", "starfish"),
            batch_size=batch_size,
            seed=self.seed,
            validation_split=0.2,
            shuffle=shuffle,
            subset="training",
            image_size=image_size,
        )

        self.validation = image_dataset_from_directory(
            os.path.join(self.input_file, "train_images_thumb"),
            labels="inferred",
            label_mode="binary",
            class_names=("background", "starfish"),
            batch_size=batch_size,
            seed=self.seed,
            validation_split=0.2,
            shuffle=shuffle,
            subset="validation",
            image_size=image_size,
        )

    def get_training(self, **kwargs):

        """
        Fetch the training set. Keyword arguments are passed down to _load_dataset.

        """

        # Cut down to thumbnails
        if not hasattr(self, "training"):
            self._load_dataset()

        return self.training

    def get_validation(self, **kwargs):

        """
        Fetch the validation set. Keyword arguments are passed down to _load_dataset.

        """

        # Cut down to thumbnails
        if not hasattr(self, "validation"):
            self._load_dataset()

        return self.validation
