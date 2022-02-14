# High-level class for the full Faster R-CNN network.
import tensorflow as tf
import tensorflow_addons as tfa
import backbone, classifier, data_utils, roi_pooling, rpn, evaluation, callback, jointmodel
import os


class FasterRCNNWrapper:
    def __init__(
        self,
        input_shape=(720, 1280, 3),
        n_proposals=10,
        positive_threshold=0.5,
        datapath="/content",
        backbone_type="ResNet50",
        backbone_weights="finetune",
        rpn_weights="train",
        rpn_kwargs={},
        roi_kwargs={},
        classifier_weights="train",
        classifier_kwargs={},
        classifier_learning_rate={
            "epochs": [1, 4, 7],
            "values": [
                1e-3,
                1e-4,
                1e-5,
            ],
        },
        classifier_weight_decay={
            "epochs": [
                1,
                4,
                7,
            ],
            "values": [
                1e-4,
                1e-5,
                1e-6,
            ],
        },
        classifier_momentum=0.9,
        classifier_clipvalue=1e1,
        classifier_augmentation={
            "zoom": 0.01,
            "rotation": 0.01,
            "gaussian": 5.0,
            "contrast": 0.25,
        },
        validation_recall_thresholds=[0.1, 0.25, 0.5, 0.75, 0.9],
        debug=1,
    ):

        """
        Instantiate the wrapper for the overall Faster R-CNN

        Arguments:

        input_shape : arr or tuple
            Shape of input image (y,x,channels)
        n_proposals : int
            Number of regions proposed by the full network. Note that
            this is (by construction) the maximum number of positive examples
            in an image that can be detected.
        positive_threshold : float
            Classification score at which to declare a region as a starfish.
        datapath : str
            Location of the competition dataset.
        backbone_type : str
            Flavor of backone to use for feature extraction. Should
            refer to  a subclass of Backbone(). Currently supported options
            are InceptionResNet-V2, VGG16, and ResNet50.
        backbone_weights : str
            Options are 'imagenet' to use pretrained weights from ImageNet, 'finetune'
            to run the fine tuning loop with a classification network on thumbnails,
            or a file path to load existing fine-tuned weights.
        rpn_weights : str
            Load pre-trained weights for the RPN from this file path, 'skip', or 'train'.
        rpn_kwargs : dict
            Optional keyword arguments passed to the RPN wrapper.
        roi_kwargs : dict
            Optional keyword arguments passed to the RoI Pooling layer.
        classifier_weights : str
            Saved weights for the final classification network, 'skip', or 'train'
        classifier_kwargs : dict
            Optional keyword arguments passed to the Classifier wrapper.
        classifier_learning_rate : dict
            {'epoch' : list of epoch number, 'rate' : list of learning rates}
        classifier_weight_decay : dict
            {'epoch' : list of epoch number, 'rate' : list of decay rates}
        classifier_momentum : float
            Momentum parameter for the SGDW optimizer.
        classifier_clipvalue : float
            Maximum allowable gradient for the SGDW optimizer.
        classifier_augmentation : dict
            Parameters to pass to the augmentation segment when training. The Gaussian noise augmentation
            and contrast are copied over from the backbone fine tuning. The translation and rotation
            should be small enough to not meaningfully break the matching of RoI to the ground truth boxes.
        debug : int
            0) Train on every image in the dataset, no validation set
            1) Run with 80% of the data and a validation set
            2) Run in debug mode with 10% of data, no validation set, and 3 epochs.
            Usually the answer is 1).
        """

        # Record for posterity
        self.input_shape = input_shape
        self.n_proposals = n_proposals
        self.positive_threshold = positive_threshold
        self.debug = debug

        # Store because we'll use this again the fine tuning loops
        self.classifier_learning_rate = classifier_learning_rate
        self.classifier_weight_decay = classifier_weight_decay
        self.classifier_momentum = classifier_momentum
        self.classifier_clipvalue = classifier_clipvalue
        self.validation_recall_thresholds = validation_recall_thresholds
        self.classifier_augmentation = classifier_augmentation

        # Check debug mode is valid
        assert isinstance(self.debug, int) and self.debug in [0, 1, 2]

        if self.debug == 0:  # No validation set, use everything we can
            self.data_kwargs = {"validation_split": 0.01}
            self.epoch_kwargs = {}
        elif self.debug == 2:  # Small dataset to check code
            self.data_kwargs = {"validation_split": 0.99}
            self.epoch_kwargs = {"epochs": 3}
        else:  # default parameters
            self.data_kwargs = {}
            self.epoch_kwargs = {}

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
        self.instantiate_classifier(
            classifier_weights, self.classifier_augmentation, classifier_kwargs
        )

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

        if not (
            rpn_weights.lower() in ["skip", "train"]
        ):  # Load the weights from a file

            assert os.path.exists(rpn_weights)
            # Run dummy data through to build the network, then load weights
            minibatch = (
                self.data_loader_full.get_training(**self.data_kwargs).__iter__().next()
            )
            self.rpnwrapper.propose_regions(minibatch[0], input_images=True)
            self.rpnwrapper.load_rpn_state(rpn_weights)
            del minibatch

        elif rpn_weights.lower() == "train":  # train the RPN with the default settings

            self.rpnwrapper.train_rpn(
                self.data_loader_full.get_training(**self.data_kwargs),
                valid_dataset=self.data_loader_full.get_validation(**self.data_kwargs)
                if self.debug == 1
                else None,
                **self.epoch_kwargs
            )

        # else: Skip RPN training

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

    def instantiate_classifier(
        self, classifier_weights, classifier_augmentation, classifier_kwargs
    ):

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
        if "epochs" in self.epoch_kwargs.keys():
            epochs = self.epoch_kwargs["epochs"]
        elif "epochs" in classifier_kwargs.keys():
            epochs = classifier_kwargs.pop("epochs")
        else:
            epochs = 9

        # Note that this is associated with self.backbone whereas
        # the rpn is associated with self.backbone_rpn
        self.classmodel = classifier.ClassifierModel(
            self.backbone,
            self.rpnwrapper,
            self.RoI_pool,
            self.data_loader_full.decode_label,
            self.n_proposals,
            classifier_augmentation,
            **classifier_kwargs
        )

        if not (classifier_weights.lower() in ["skip", "train"]):

            # Run dummy data through the network and then copy in weights
            assert os.path.exists(classifier_weights)
            minibatch = (
                self.data_loader_full.get_training(**self.data_kwargs).__iter__().next()
            )
            features = self.backbone(minibatch[0])
            features, roi = self.RoI_pool(
                (
                    features,
                    self.rpnwrapper.propose_regions(
                        features, input_images=False, output_images=False
                    ),
                )
            )
            self.classmodel.call((features, roi))
            del minibatch

            self.classmodel.load_classifier_state(classifier_weights)

        elif (
            classifier_weights.lower() == "train"
        ):  # Do the first order training of the classification weights

            self.class_optimizer = tfa.optimizers.SGDW(
                learning_rate=self.classifier_learning_rate["values"][0],
                weight_decay=self.classifier_weight_decay["values"][0],
                momentum=self.classifier_momentum,
                clipvalue=self.classifier_clipvalue,
            )
            self.class_metrics = (
                [
                    evaluation.ThresholdRecall(
                        _threshold,
                        self.data_loader_full.decode_label,
                        name="recall_score_%.2d" % _threshold,
                    )
                    for _threshold in self.validation_recall_thresholds
                ],
            )

            self.classmodel.compile(
                optimizer=self.class_optimizer, metrics=self.class_metrics
            )

            self.classmodel.fit(
                self.data_loader_full.get_training(**self.data_kwargs),
                epochs=epochs,
                validation_data=self.data_loader_full.get_validation(**self.data_kwargs)
                if self.debug == 1
                else None,
                callbacks=[
                    callback.LearningRateCallback(
                        self.classifier_learning_rate, self.classifier_weight_decay
                    )
                ],
            )

        # else: Skip classifier training

    def predict(self, images, return_mode="string"):
        """
        Make predictions for an image.

        Arguments:

        Image : tf.tensor
            Minibatch of image(s) to register a prediction for.
        return_mode : str
            String encoding how to return the results. Supported modes
            are either 'dict' or 'string' returning a python dict
            or the string formatting expected by the test suite.

        """

        assert return_mode.lower() in ["string", "dict"]

        # Forward mode
        features = self.backbone(images)
        features, roi = self.RoI_pool(
            (
                features,
                self.rpnwrapper.propose_regions(
                    features, input_images=False, output_images=False
                ),
            )
        )

        # Unstack into individual coordinates
        x, y, w, h, score = tf.unstack(self.classmodel.call((features, roi)), axis=-1)

        # Convert to a python dictionary
        minibatch_regions = [
            [
                {
                    "x": x[i_image, i_roi].numpy(),
                    "y": y[i_image, i_roi].numpy(),
                    "width": w[i_image, i_roi].numpy(),
                    "height": h[i_image, i_roi].numpy(),
                    "score": score[i_image, i_roi].numpy(),
                }
                for i_roi in range(self.n_proposals)
            ]
            for i_image in range(score.shape[0])
        ]

        # Trim the outputs
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
            if return_mode.lower() == "dict":
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

    # Convert the output string to the format expected by the test set evaluation
    # From code other people have posted the order seems to be (score, x, y, w, h)
    @staticmethod
    def _region_to_string(region):
        return "%.2f %d %d %d %d" % (
            region["score"],
            int(region["y"]),
            int(region["x"]),
            int(region["width"]),
            int(region["height"]),
        )

    def do_fine_tuning(
        self, epochs, learning_rate=1e-5, weight_decay=1e-7, momentum=0.9, clipvalue=1e1
    ):
        """
        Free up the backbone and run a joint training routine.

        The tensorflow model building paradigm has pushed us into
        a bit of a weird corner. We will instantiate a temporary
        class whose job is to update the weights of all three components
        during a fine tuning loop but otherwise has an empty call() method.

        epochs : int
            If > 0 run this many epochs in fine tuning mode where the backbone weights are
            freed. Uses the learning rate and weight decay from the final epoch.
        learning_rate : float
            Learning rate to be used for both updating the backbone + RPN and classifier.
        weight_decay : float
            Weight decay parameter in the optimizer.
        momentum : float
            Momentum parameter in the optimizer.
        clipvalue : float
            Gradient clipping in the optimizer.

        Note that the optimizer parameters can be set exactly once, changing them subsequently
        has no impact.

        """

        # Short circuit if there is nothing to do.
        if epochs == 0:
            return

        # Sanity checking
        if hasattr(self, "class_optimizer"):
            raise ValueError(
                "Cannot recompile a model with a new optimizer, reinstantiate the overall FasterRCNNWrapper()."
            )

        # Instantiate the joint model
        # and optimizer classes

        if not hasattr(self, "joint_model"):

            self.joint_model = jointmodel.JointModel(
                self.backbone,
                self.rpnwrapper.rpnmodel,
                self.data_loader_full.decode_label,
                self.rpnwrapper.rpnmodel.augmentation,
            )
            self.joint_optimizer = tfa.optimizers.SGDW(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
                clipvalue=clipvalue,
            )

            self.joint_metrics = [
                evaluation.TopNRegionsRecall(
                    self.rpnwrapper.top_n_recall,
                    self.data_loader_full.decode_label,
                    name="top%d_recall" % self.rpnwrapper.top_n_recall,
                )
            ]

            self.joint_model.compile(
                optimizer=self.joint_optimizer, metrics=self.joint_metrics
            )

        if not hasattr(self, "class_optimizer_fine"):

            self.class_optimizer_fine = tfa.optimizers.SGDW(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
                clipvalue=clipvalue,
            )

            self.class_metrics_fine = [
                evaluation.ThresholdRecall(
                    _threshold,
                    self.data_loader_full.decode_label,
                    name="recall_score_%.2d" % _threshold,
                )
                for _threshold in self.validation_recall_thresholds
            ]

            self.classmodel.compile(
                optimizer=self.class_optimizer_fine, metrics=self.class_metrics_fine
            )

        ###########
        #
        # First train the RPN + backbone in isolation
        #
        ###########

        # Important - free up the backbone weights
        self.backbone.set_trainable(True)

        # Compile the joint model using the fine runing optimizer and
        # the same metrics as the classifier

        # Run the model fit, like the classifier but with no callbacks because we never touch the learning rate
        self.joint_model.fit(
            self.data_loader_full.get_training(**self.data_kwargs),
            epochs=epochs,
            validation_data=self.data_loader_full.get_validation(**self.data_kwargs)
            if self.debug == 1
            else None,
        )

        # Set the backbone where we found it
        self.backbone.set_trainable(False)

        ###########
        #
        # Now retrain the classifier
        #
        ###########

        self.classmodel.fit(
            self.data_loader_full.get_training(**self.data_kwargs),
            epochs=epochs,
            validation_data=self.data_loader_full.get_validation(**self.data_kwargs)
            if self.debug == 1
            else None,
        )
