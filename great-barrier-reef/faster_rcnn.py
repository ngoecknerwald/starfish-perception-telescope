# This script will contain the faster R-CNN class
import tensorflow as tf
import rpn, backbone, data_utils


# OK, so here is Neil's TODO list.
#
# b) Write up the ROI pooling operation from the region proposals. Basic idea: define an ROI pool size, then
# take the bbox from the RPN into feature coords, divide up into nxn subregions, take max or average pool in those
# regions -> Pass to the convolutional layers at the end. See if TF has any builtin support for ROI pooling. Use a
# free parameter for NMS from the RPN layer.
#
# c) After ROI pool, connect up the fully connected layers.
#
# d) Let's make a method to save the state of the RPN


class FasterRCNNWrapper:
    def __init__(
        self,
        backbone_type='InceptionResNet-V2',
        input_shape=(720, 1280, 3),
        backbone_weights=None,
        data_loader_thumb=None,
        rpn_kwargs={},
        rpn_weights=None,
        n_proposals=10,
    ):

        '''
        Instantiate the wrapper for the overall Faster R-CNN

        Arguments:

        backbone_type : str
            Flavor of backone to use for feature extraction. Should
            refer to  a subclass of Backbone(). Currently supported options
            are InceptionResNet-V2 and VGG16.
        input_shape : tuple
            Shape of the input images.
        backbone_weights : str or None
            Options are 'imagenet' to use pretrained weights from ImageNet, 'finetune'
            to run the fine tuning loop with a classification network on thumbnails,
            or a file path to load fine-tuned weights from a file. None gives a
            random weight initialization.
        data_loader_thumb : data_utils.DataLoaderThumbnail() or None
            Thumbnail data loader required if backbone_weights == 'pretrain'
            used to pretrain the backbone.
        rpn_kwargs : {}
            Arguments to pass to the RPN, usually some set of hyperparameters.
        rpn_weights : str or None
            Load pre-trained RPN weights from a file or None.
        n_proposals : int
            Number of proposals made by the RPN.
        '''

        # Input scrubbing
        if backbone_weights is None:
            print('Returning a randomly initialized backbone!')
        else:
            backbone_weights = backbone_weights.lower()

        # Record for posterity
        self.input_shape = input_shape

        ########
        #
        # Instantiate backbone
        #
        ########

        if backbone.lower() == 'inceptionresnet-v2':
            self.backbone = backbone.Backbone_InceptionResNetV2(
                input_shape=input_shape,
                weights=backbone_weights if backbone_weights == 'imagenet' else None,
            )
        elif backbone.lower() == 'vgg16':
            self.backbone = backbone.BackBone_VGG16(
                input_shape=input_shape,
                weights=backbone_weights if backbone_weights == 'imagenet' else None,
            )
        else:
            raise ValueError('Argument backbone=%s is not recognized.' % backbone)

        # Now pretrain the backbone or load weights if requested
        if backbone_weights == 'pretrain':

            # Instantiate a temporary backbone
            if backbone.lower() == 'inceptionresnet-v2':
                spine = backbone.Backbone_InceptionResNetV2(
                    weights='imagenet', input_shape=None
                )
            elif backbone.lower() == 'vgg16':
                spine = backbone.BackBone_VGG16(weights='imagenet', input_shape=None)

            # Data loading
            assert isinstance(data_loader_thumb, data_utils.DataLoaderThumbnail)
            train_data = data_loader_thumb.get_training(
                validation_split=0.2, batch_size=64, shuffle=True
            )
            valid_data = data_loader_thumb.get_validation(
                validation_split=0.2, batch_size=64, shuffle=True
            )

            # Train the temporary backbone, then copy weights over
            spine.pretrain(train_data, validation_data=valid_data)
            backbone.network.set_weights(spine.network.get_weights())

            # Clean up
            del spine

        # Load the backbone weights
        elif backbone not in ['imagenet', None] and os.path.exists(backbone_weights):
            print('Loading backbone weights from %s' % backbone_weights)
            self.backbone.load_backbone(backbone_weights)

        # Leave the imagenet weights as is
        else:
            pass

        # Instantiate the RPN wrapper
        self.rpn = RPNWrapper(backbone, **rpn_kwargs)

        if rpn_weights is not None:
            self.rpn.rpn = tf.saved_model.load(rpn_weights)

        self.n_proposals = n_proposals
        self.rcnn = FasterRCNN(self.n_proposals)

    def train_RPN(self, data_loader_full):
        '''
        Train the RPN itself.

        Arguments:

        data_loader_full : data_utils.DataLoaderThumbnail()
            Wrapper for interfacing with the dataset. Required
            to train the RPN or final network.

        '''

        pass
