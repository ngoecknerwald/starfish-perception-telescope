import tensorflow as tf


class JointModel(tf.keras.Model):
    def __init__(self, backbone, rpnmodel, roi_pool, classifier):

        super().__init__()

        # Store the components that we will need for the train and step steps.
        self.backbone = backbone
        self.rpnmodel = rpnmodel
        self.roi_pool = self.roi_pool
        self.classifier = classifier

    def __del__(self):
        """
        Dummy method to ensure that the constituent compoents
        of the model aren't deleted because we still need them.
        This model class has a somewhat transient existence.

        """

        pass

    def call(self, data):
        """
        Dummy method, the actual prediction methods
        live in faster_rcnn.py. This is to ensure
        that nothing is done here.

        """

        return None

    def train_step(self, data):

        images, labels = data

        pass

    def test_step(self, data):

        images, labels = data

        pass

    # Finally, we want to instantiate a copy of the backbone that the
    # RPN will use to propose regions. We will end up fine tuning the
    # backbone in conjunction with the classification layers and we
    # don't want the backbone changing under the RPN's feet.

    # Create a new backbone and copy weights over to make a deep copy
    # init_args = {"input_shape": None, "weights": None}
    # self.backbone_rpn = backbone.instantiate(backbone_type, init_args)
    # self.backbone_rpn.network.set_weights(self.backbone.network.get_weights())

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
