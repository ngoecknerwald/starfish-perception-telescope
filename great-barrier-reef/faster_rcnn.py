# This script will contain the faster R-CNN class

import rpn, backbone


class FasterRCNN:
    def __init__(self, backbone='InceptionResNet-v2', rpn_kwargs={}):

        '''
        Instantiate the wrapper for the overall Faster R-CNN

        backbone : str
            Flavor of backone to use for feature extraction. Should
            refer to  a subclass of Backbone()
        rpn_kwargs : {}
            Arguments to pass to the RPN, usually some set of hyperparameters.

        '''

        # Instantiate components
        if backbone.lower() == 'inceptionresnet-v2':
            self.backbone = Backbone_InceptionResNetV2()
        else:
            raise ValueError('Argument backbone=%s is not recognized.')

        # Instantiate the RPN wrapper
        self.rpn = RPNWrapper(**rpn_kwargs)

    def train(self, images_full, images_thumb):
        pass
