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
        positive_threshold=0.5,
        datapath="/content",
        backbone_type="InceptionResNet-V2",
        backbone_weights="finetune",
        rpn_weights=None,
        rpn_kwargs={},
        roi_kwargs={},
        classifier_weights=None,
        classifier_kwargs={},
        finetuning_epochs=5,
        learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[
                10000,
            ],
            values=[1e-3, 1e-4],
        ),
        weight_decay=tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[
                10000,
            ],
            values=[1e-4, 1e-5],
        ),
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
        finetuning_epochs : int
            Number of epochs to fine-tune the backbone, RPN, and classifier.

        """

        # Record for posterity
        self.input_shape = input_shape
        self.n_proposals = n_proposals
        self.positive_threshold = positive_threshold

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

        # Optimizer for the full network training
        self.optimizer = tfa.optimizers.SGDW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
            clipvalue=1e2,
        )

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

        # TODO enable this with the fine tuning loop.

        # Finally, we want to instantiate a copy of the backbone that the
        # RPN will use to propose regions. We will end up fine tuning the
        # backbone in conjunction with the classification layers and we
        # don't want the backbone changing under the RPN's feet.

        # Create a new backbone and copy weights over to make a deep copy
        # init_args = {"input_shape": None, "weights": None}
        # self.backbone_rpn = backbone.instantiate(backbone_type, init_args)
        # self.backbone_rpn.network.set_weights(self.backbone.network.get_weights())

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
        self.rpnwrapper = rpn.RPNWrapper(
            self.backbone, self.data_loader_full.decode_label, **rpn_kwargs
        )

        if rpn_weights is not None:  # Load the weights from a file

            assert os.path.exists(rpn_weights)
            # Run dummy data through to build the network, then load weights
            minibatch = self.data_loader_full.get_training().__iter__().next()
            self.rpnwrapper.propose_regions(minibatch[0], is_images=True)
            self.rpnwrapper.load_rpn_state(rpn_weights)
            del minibatch

        else:  # train the RPN with the default settings

            self.rpnwrapper.train_rpn(
                self.data_loader_full.get_training(),
            )

    def instantiate_RoI_pool(self, roi_kwargs):

        """
        Create the RoI pooling operation.

        Arguments:

        roi_kwargs : dict
            Optional keyword arguments to pass to the RoIPooling constructor.

        """

        self.RoI_pool = roi_pooling.RoIPooling(
            (
                int(self.backbone._output_shape[0]),
                int(self.backbone._output_shape[1]),
                int(self.backbone._output_shape[2]),
            ),
            self.n_proposals,
            **roi_kwargs
        )

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
        self.classmodel = classifier.ClassifierModel(
            self.backbone,
            self.rpnwrapper,
            self.RoI_pool,
            self.data_loader_full.decode_label,
            self.n_proposals,
            **classifier_kwargs
        )

        if classifier_weights is not None:

            assert os.path.exists(classifier_weights)
            self.classmodel.load_classifier_state(classifier_weights)

        else:  # Do the first order training of the classification weights

            self.train_classifier(epochs)

    def train_classifier(self, epochs, kwargs={}):
        """
        Main training loop iterating over a dataset.

        Arguments:

        train_dataset: tensorflow dataset
            Dataset of input images. Minibatch size and validation
            split are determined when this is created.
        epochs : int
            Number of epochs to run training for.

        """
        training = self.data_loader_full.get_training()

        self.classmodel.compile(
            optimizer=self.optimizer,
        )
        self.classmodel.fit(training, epochs=epochs, **kwargs)

    # TODO: make this function compatible with the new classifier architecture
    def predict(self, image, return_dict=False):

        """
        Make predictions for an image.

        Arguments:

        Image : tf.tensor
            Minibatch of image(s) to register a prediction for.
        """
        print("Function not currently compatible with classifier architecture")
        assert False

        # Usual invocation, taking advantage of the shared backbone
        features = self.backbone.extractor(image)
        roi = self.rpnwrapper.propose_regions(features)
        features, roi = self.RoI_pool(features, roi)

        # Run the classifier in forward mode
        minibatch_regions = self.classmodel(
            features,
            roi.astype("float32"),
        )

        # output munging
        minibatch_return = []
        for regions in minibatch_regions:

            # Clip regions
            _regions = [
                region
                for region in regions
                if region["score"] > self.positive_threshold
            ]
            # Sort by objectness
            _regions = sorted(
                _regions, key=lambda region: region["score"], reverse=True
            )
            if return_dict:
                minibatch_return.append(_regions)
            else:
                minibatch_return.append(
                    " ".join(
                        [
                            FasterRCNNWrapper._region_to_string(region)
                            for region in _regions
                        ]
                    )
                )

        return minibatch_return

    # We need to figure out if the correct order is what we're calling (x, y) or (y, x)
    # because the documentation on the website is unclear about this
    #
    # From code other people have posted it seems to be (x, y, w, h) so I'm leaving that for now
    @staticmethod
    def _region_to_string(region):
        return "%02f %d %d %d %d" % (
            region["score"],
            int(region["x"]),
            int(region["y"]),
            int(region["width"]),
            int(region["height"]),
        )

    def fine_tuning_loop(self):

        pass

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
        #        self.classmodel.train_step(
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
