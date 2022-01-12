# High-level class for the full Faster R-CNN network.
import tensorflow as tf
import backbone, classifier, data_utils, rpn, roi_pooling
import os


class FasterRCNN(tf.keras.Model):
    def __init__(self, backbone, rpn, roi_pooling, classifier):
        super().__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_pooling = roi_pooling
        self.classifier = classifier

    def call(self, x):

        # Note from Neil: Do we need / want to do separate backbone
        # feature extraction for the RPN and the final network?
        # One could imagine some tuning operation where we run tf.keras.Model.compile()
        # and Model.fit() allowing that to adjust the backbone but don't want the
        # weights changing under the RPN's feet. That might be an argument for
        # having two backbone instances and just manually propagating updates in the training
        # cadence with get_weights() and set_weights().

        # Also Note from Neil, if we don't do that then we should avoid double
        # feature extraction running the backbone independently between
        # the rpn.propose_regions() call and directly as part of the final network.

        # Otherwise this looks dank
        roi = self.rpn.propose_regions(x)
        x = self.backbone.extractor(x)
        x = self.roi_pooling((x, roi))
        cls, bbox = self.classifier.classifier(x)
        return cls, bbox


class FasterRCNNWrapper:
    def __init__(
        self,
        input_shape=(720, 1280, 3),
        n_proposals=30,
        datapath="/content",
        backbone_type="InceptionResNet-V2",
        backbone_weights="finetune",
        backbone_trainable=False,
        rpn_weights=None,
        rpn_kwargs={},
        rpn_trainable=False,
        roi_kwargs={},
        classifier_weights=None,
        classifier_kwargs={},
    ):

        """
        Instantiate the wrapper for the overall Faster R-CNN

        Arguments:
        input_shape : arr or tuple
            Shape of input image (y,x,channels)
        n_proposals : 
            Number of regions proposed by the RPN.
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
        backbone_trainable : bool
            Train/freeze backbone weights.
        rpn_weights : str or None
            Load pre-trained weights for the RPN from this file path.
        rpn_kwargs : dict
            Optional keyword arguments passed to the RPN wrapper.
        rpn_trainable : bool
            Train/freeze rpn weights.
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
        self._backbone_trainable = backbone_trainable
        self.instantiate_backbone(backbone_type, backbone_weights)

        # Instantiate the RPN
        self._rpn_trainable = rpn_trainable
        self.instantiate_RPN(rpn_weights, rpn_kwargs)

        # Instantiate the tail network
        self.instantiate_RoI_pool(roi_kwargs)
        self.instantiate_classifier(classifier_weights, classifier_kwargs)

        self.instantiate_FasterRCNN()

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

    @property
    def backbone_trainable(self):
        return self._backbone_trainable

    @backbone_trainable.setter
    def backbone_trainable(self, value):
        self._backbone_trainable = value
        self.backbone.extractor.trainable = value

    @property
    def rpn_trainable(self):
        return self._rpn_trainable

    @rpn_trainable.setter
    def rpn_trainable(self, value):
        self._rpn_trainable = value
        self.rpnwrapper.rpn.trainable = value
    
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

        self.backbone.extractor.trainable = self.backbone_trainable

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
        
        self.rpnwrapper.rpn.trainable = self.rpn_trainable

    def instantiate_RoI_pool(self, roi_kwargs):

        """
        Do whatever constructor-y things need to be done for the NMS and ROI pooling operations.

        """

        self.RoI_pool = roi_pooling.ROIPooling(self.n_proposals, **roi_kwargs)

    def instantiate_classifier(self, classifier_weights, classifier_kwargs):

        """
        Instantiate the classifier wrapper.

        Arguments:

        classifier_weights : str or None
            Saved weights for the final classification network.
        """

        # TODO wire the input sizes from the backbone and IoU suppression / RoI pooling into here.
        self.classwrapper = classifier.ClassifierWrapper(self.n_proposals, **classifier_kwargs)

        if classifier_weights is not None:

            assert os.path.exists(classifier_weights)
            self.classwrapper.load_classifier_state(classifier_weights)

    def instantiate_FasterRCNN(self):
        self.FasterRCNN = FasterRCNN(
            self.backbone, self.rpnwrapper, self.RoI_pool, self.classwrapper
        )

    def run_training(self):
        """
        Train the whole shebang including the classifier

        """
