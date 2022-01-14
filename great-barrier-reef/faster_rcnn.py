# High-level class for the full Faster R-CNN network.
import tensorflow as tf
import tensorflow_addons as tfa
import backbone, classifier, data_utils, roi_pooling, rpn
import os


class FasterRCNNWrapper:
    def __init__(
        self,
        input_shape=(720, 1280, 3),
        n_proposals=10,
        datapath="/content",
        backbone_type="InceptionResNet-V2",
        backbone_weights="finetune",
        rpn_weights=None,
        rpn_kwargs={},
        roi_kwargs={},
        classifier_weights=None,
        classifier_kwargs={},
        finetuning_epochs=5,
    ):

        """
        Instantiate the wrapper for the overall Faster R-CNN

        Arguments:
        input_shape : arr or tuple
            Shape of input image (y,x,channels)
        n_proposals :
            Number of regions proposed by the full network. Note that
            this is (by construction) the maximum number of positive examples
            in an image that can be detected.
        datapath : str
            Location of the competition dataset.
        backbone_type : str
            Flavor of backone to use for feature extraction. Should
            refer to  a subclass of Backbone(). Currently supported options
            are InceptionResNet-V2 and VGG16.
        backbone_weights : str
            Options are 'imagenet' to use pretrained weights from ImageNet, 'finetune'
            to run the fine tuning loop with a classification network on thumbnails,
            or a file path to load existing fine-tuned weights.
        rpn_weights : str or None
            Load pre-trained weights for the RPN from this file path.
        rpn_kwargs : dict
            Optional keyword arguments passed to the RPN wrapper.
        roi_kwargs : dict
            Optional keyword arguments passed to the RoI Pooling layer.
        classifier_weights : str or None
            Saved weights for the final classification network.
        classifier_kwargs : dict
            Optional keyword arguments passed to the Classifier wrapper.

        """

        # Record for posterity
        self.input_shape = input_shape
        self.n_proposals = n_proposals

        # Instantiate data loading class
        self.instantiate_data_loaders(
            datapath, do_thumbnail=(backbone_weights == "finetune")
        )

        # Instantiate backbone
        self.instantiate_backbone(backbone_type, backbone_weights)

        # Instantiate the RPN
        self.instantiate_RPN(rpn_weights, rpn_kwargs)

        # Instantiate the tail network
        self.instantiate_RoI_pool(roi_kwargs)

        # This should be instantiated last
        self.instantiate_classifier(classifier_weights, classifier_kwargs)

    def instantiate_data_loaders(self, datapath, do_thumbnail=False):
        """
        Create the data loader classes.

        Arguments:

        datapath : str
            Location of the competition dataset.
        do_thumbnail : bool
            Also create the thumbnails for backbone pretraining.
        """

        self.data_loader_full = data_utils.DataLoaderFull(input_file=datapath)

        if do_thumbnail:
            self.data_loader_thumb = data_utils.DataLoaderThumbnail(input_file=datapath)
        else:
            self.data_loader_thumb = None

    def instantiate_backbone(
        self,
        backbone_type,
        backbone_weights,
    ):
        """
        Instantiate (and pretrain) the backbone.

        Arguments:

        backbone_type : str
            Flavor of backone to use for feature extraction. Should
            refer to  a subclass of Backbone(). Currently supported options
            are InceptionResNet-V2 and VGG16.
        backbone_weights : str
            Options are 'imagenet' to use pretrained weights from ImageNet, 'finetune'
            to run the fine tuning loop with a classification network on thumbnails,
            or a file path to load fine-tuned weights from a file.

        """

        # Input scrubbing
        if backbone_weights.lower() == "finetune" and self.data_loader_thumb is None:
            raise ValueError("Thumbnail loader class needed to finetune the backbone.")

        # Figure out what backbone type we are dealing with here and create it,
        # note that the weights are set to random unless specifically set to imagenet
        init_args = {
            "input_shape": self.input_shape,
            "weights": "imagenet" if backbone_weights.lower() == "imagenet" else None,
        }
        self.backbone = backbone.instantiate(backbone_type, init_args)

        # Load or finetune the weights if requested
        if backbone_weights.lower() == "imagenet":  # Done, no need to do anything else
            pass

        elif backbone_weights.lower() == "finetune":  # Load weights from a file
            # Let the backbone for finetuning infer the thumbnail shape on the fly
            init_args = {"input_shape": None, "weights": "imagenet"}
            spine = backbone.instantiate(backbone_type, init_args)

            # Data loading
            assert isinstance(self.data_loader_thumb, data_utils.DataLoaderThumbnail)
            train_data = self.data_loader_thumb.get_training()
            valid_data = self.data_loader_thumb.get_validation()

            # Train the temporary backbone
            spine.pretrain(train_data, validation_data=valid_data)

            # Copy the convolutional weights over
            self.backbone.network.set_weights(spine.network.get_weights())

            # Clean up
            del spine

        else:  # Load weights
            assert os.path.exists(backbone_weights)
            print("Loading backbone weights from %s" % backbone_weights)
            self.backbone.load_backbone(backbone_weights)

        # Finally, we want to instantiate a copy of the backbone that the
        # RPN will use to propose regions. We will end up fine tuning the
        # backbone in conjunction with the classification layers and we
        # don't want the backbone changing under the RPN's feet.

        # Create a new backbone and copy weights over to make a deep copy
        init_args = {"input_shape": None, "weights": None}
        self.backbone_rpn = backbone.instantiate(backbone_type, init_args)
        self.backbone_rpn.network.set_weights(self.backbone.network.get_weights())

    def instantiate_RPN(self, rpn_weights, rpn_kwargs):
        """
        Train the RPN itself.

        Arguments:

        rpn_weights : str
            Load pre-trained weights for the RPN from this file path.
        rpn_kwargs : dict
            Optional keyword arguments passed to the RPN wrapper.

        """

        # Create the RPN wrapper
        self.rpnwrapper = rpn.RPNWrapper(self.backbone, **rpn_kwargs)

        if rpn_weights is not None:  # Load the weights from a file

            assert os.path.exists(rpn_weights)
            self.rpnwrapper.load_rpn_state(rpn_weights)

        else:  # train the RPN with the default settings

            self.rpnwrapper.train_rpn(
                self.data_loader_full.get_training(), self.data_loader_full.decode_label
            )

    def instantiate_RoI_pool(self, roi_kwargs):

        """
        Create the RoI pooling operation.

        Arguments:

        roi_kwargs : dict
            Optional keyword arguments to pass to the RoIPooling constructor.

        """

        self.RoI_pool = roi_pooling.RoIPooling(self.backbone.output_shape, **roi_kwargs)

    def instantiate_classifier(self, classifier_weights, classifier_kwargs):

        """
        Instantiate the classifier wrapper.

        Arguments:

        classifier_weights : str or None
            Saved weights for the final classification network.
        classifier_kwargs : dict
            Optional keyword arguments to pass to the classification constructor.
            Takes 'epochs' : int to set the number of epochs to run the initial training.
        """

        # Number of epochs to train the classifier network
        epochs = classifier_kwargs.pop("epochs", 5)

        # Note that this is associated with self.backbone whereas
        # the rpn is associated with self.backbone_rpn
        self.classwrapper = classifier.ClassifierWrapper(
            self.backbone, self.n_proposals, **classifier_kwargs
        )

        if classifier_weights is not None:

            assert os.path.exists(classifier_weights)
            self.classwrapper.load_classifier_state(classifier_weights)

        else:  # Do the first order training of the classification weights

            self.train_classifier(epochs)

    def train_classifier(self, epochs):
        """
        Train the classifier holding the backbone weights fixed. Written
        as a method in FasterRCNNWrapper because the training step needs
        access to the backbone, RPN, and RoI pooling layers.

        Arguments:

        epochs : int
            Number of epochs to run training for.

        This method does not fine tune the backbone weights.

        """

        # This does a forward pass through the RPN
        # and hands the proposed regions + features to the classifier
        for epoch in range(epochs):

            print("Classifier training epoch %d" % epoch, end="")

            for i, (train_x, label_x) in enumerate(
                self.data_loader_full.get_training()
            ):

                if i % 100 == 0:
                    print(".", end="")

                # Propose regions and compute features with the
                # backbone associated with the classifier
                roi = self.rpnwrapper.propose_regions(train_x)
                features = self.backbone.extractor(train_x)

                # Clip the RoI and pool the features
                features, roi = self.RoI_pool(features, roi)

                # Take a gradient step, TODO hand images down instead of features
                self.classwrapper.training_step(
                    features,
                    roi,
                    [self.data_loader_full.decode_label(_label) for _label in label_x],
                    update_backbone=False,
                )

            print("")

    def fine_tuning_loop(self, epochs=5):
        """
        Run a fine tuning loop.


        """

        # Pseudocode

        # Set self.backbone_rpn trainable=False <- the copy of the backbone fed into the RPN
        # Set self.backbone trainable = True <- the copy of the backbone fed into the classifier

        # for i in range(epochs):
        #
        #    # Fine tuning loop for the backbone + classifier
        #    for i, (train_x, label_x) in enumerate(
        #        self.data_loader_full.get_training()
        #    ):
        #
        #        # forward mode
        #        roi = self.rpnwrapper.propose_regions(train_x)
        #        features = self.backbone.extractor(train_x)
        #        features, roi = self.RoI_pool(features, roi)

        #        # TODO we need to move the backbone call into the training step method for this to work
        #        # TODO we also need to define "fine tuning" learning rates
        #        self.classwrapper.training_step(
        #            features,
        #            roi,
        #            [self.data_loader_full.decode_label(_label) for _label in label_x],
        #            update_backbone=True,
        #            fine_tuning=True
        #        )
        #
        #
        #     # Now copy over the improved backbone weights to the RPN
        #     self.bacbone_rpn.network.set_weights(self.backbone.network.get_weights)
        #
        #     # TODO add a fine_tuning kwarg to the rpn training
        #     self.rpnwrapper.train_rpn(
        #        self.data_loader_full.get_training(), self.data_loader_full.decode_label, fine_tuning=True
        #     )

        pass

    def predict(self, image, threshold):

        """
        Make predictions for an image.

        """
        return self.classwrapper.predict_classes(
            image, self.rpnwrapper.propose_regions(image), positive_thresh=0.5
        )
