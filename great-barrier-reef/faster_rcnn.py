# High-level class for the full Faster R-CNN network.
import tensorflow as tf
import rpn, backbone, data_utils


class FasterRCNNWrapper:
    def __init__(
        self,
        backbone_type='InceptionResNet-V2',
        input_shape=(720, 1280, 3),
        backbone_weights='pretrain',
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
        backbone_weights : str
            Options are 'imagenet' to use pretrained weights from ImageNet, 'finetune'
            to run the fine tuning loop with a classification network on thumbnails,
            or a file path to load fine-tuned weights from a file.
        '''

        # Record for posterity
        self.input_shape = input_shape

        # Instantiate backbone
        self.instantiate_backbone(
            backbone_type, input_shape, backbone_weights, data_loader_thumb
        )

    def instantiate_backbone(
        self, backbone_type, input_shape, backbone_weights, data_loader_thumb
    ):
        '''
        Instantiate (and pretrain) the backbone.

        Arguments:

        backbone_type : str
            Flavor of backone to use for feature extraction. Should
            refer to  a subclass of Backbone(). Currently supported options
            are InceptionResNet-V2 and VGG16.
        input_shape : tuple
            Shape of the input images.
        backbone_weights : str
            Options are 'imagenet' to use pretrained weights from ImageNet, 'finetune'
            to run the fine tuning loop with a classification network on thumbnails,
            or a file path to load fine-tuned weights from a file.
        data_loader_thumb : data_utils.DataLoaderThumbnail() or None
            Thumbnail data loader required if backbone_weights == 'pretrain'
            used to pretrain the backbone.

        '''

        # Input scrubbing
        backbone_weights = backbone_weights.lower()
        if backbone_weights == 'finetune' and data_loader_thumb is None:
            raise ValueError('Thumbnail loader class needed to finetune the backbone.')

        # Figure out what backbone type we are dealing with here and create it,
        # note that the weights are set to random unless specifically set to imagenet
        init_args = {
            'input_shape': input_shape,
            'weights': 'imagenet' if backbone_weights == 'imagenet' else None,
        }
        self.backbone = backbone.instantiate(backbone_type, init_args)

        # Load or finetune the weights if requested
        if backbone_weights == 'imagenet':  # Done, no need to do anything else
            pass

        elif backbone_weights == 'finetune':  # Load weights from a file
            # Let the backbone for finetuning infer the thumbnail shape on the fly
            init_args = {'input_shape': None, 'weights': 'imagenet'}
            spine = backbone.instantiate(backbone_type, init_args)

            # Data loading
            assert isinstance(data_loader_thumb, data_utils.DataLoaderThumbnail)
            train_data = data_loader_thumb.get_training(
                validation_split=0.2, batch_size=64, shuffle=True
            )
            valid_data = data_loader_thumb.get_validation(
                validation_split=0.2, batch_size=64, shuffle=True
            )

            # Train the temporary backbone
            spine.pretrain(train_data, validation_data=valid_data)

            # Copy the convolutional weights over
            backbone.network.set_weights(spine.network.get_weights())

            # Clean up
            del spine

        else:  # Load weights
            assert os.path.exists(backbone_weights)
            print('Loading backbone weights from %s' % backbone_weights)
            self.backbone.load_backbone(backbone_weights)

    def instantiate_RPN(self, data_loader_full):
        '''
        Train the RPN itself.

        Arguments:

        data_loader_full : data_utils.DataLoaderThumbnail()
            Wrapper for interfacing with the dataset. Required
            to train the RPN or final network.

        '''
