import tensorflow as tf


class JointModel(tf.keras.Model):
    def __init__(
        self,
        backbone,
        rpnmodel,
        roi_pool,
        classmodel,
        label_decoder,
        augmentation_params,
    ):
        """
        Dummy model for joint training. Accepts the minimal set of components
        to make this training work.

        Arguments:

        backbone : backbone.Backbone() subclass
            Feature extraction backbone used in the Faster R-CNN.
        rpnmodel : rpn.RPNModel()
            RPN model, not to be confused with the RPNWrapper() class.
        roi_pool : roi_pool.RoIPooling()
            RoI pooling and IoU supression class.
        classifier : classifier.ClassifierModel()
            Model containing the classification stages.
        label_decoder : DataLoader.decode_label()
            Callable to decode an int label and return annotations.
        augmentation_params : dict
            Parameters to augment the input images. See the Classifier class.

        """

        super().__init__()

        # Store the components that we will need for the train and step steps.
        self.backbone = backbone
        self.rpnmodel = rpnmodel
        self.roi_pool = roi_pool
        self.classmodel = classmodel
        self.label_decoder = label_decoder
        self.augmentation_params = augmentation_params

        self.augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomZoom(self.augmentation_params["zoom"]),
                tf.keras.layers.RandomRotation(self.augmentation_params["rotation"]),
                tf.keras.layers.GaussianNoise(self.augmentation_params["gaussian"]),
                tf.keras.layers.RandomContrast(self.augmentation_params["contrast"]),
            ]
        )

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

        pass

    def train_step(self, data):

        """
        Run a model training step and return metrics and loss.
        This function is a bit of a kludge in that it calls the
        feature extraction twice. I can't think of a good way around this.

        Arguments:

        data : [tf.Tensor, int]
            Minibatch of data from the full image data loading classes.

        """

        # Run the feature extractor and decode labels outside the gradient tape
        features_init = self.backbone(data[0])
        labels = self.label_decoder(data[1])

        # First we must compute the RoI. This is taken
        # as a fixed value for the classifier training
        # and therefore must live outside the gradient tape.
        roi_init = self.rpnmodel(features_init, input_images=False, output_images=False)

        # Clip and NMS the RoI, taken out of the class to avoid unnecessary
        # clipping of the feature map we won't use anyways
        roi_clip = self.roi_pool._clip_RoI(roi_init)
        roi = tf.map_fn(self.roi_pool._IoU_suppression, roi_clip)

        # Next we must accumulate RoI based on the label for the RPN loss.
        # This also should to live outside the GradientTape().
        rpn_roi = tf.map_fn(self.rpnmodel._accumulate_roi, labels)

        # Finally augment the data before taking gradients
        data_aug = self.augmentation(data[0])

        with tf.GradientTape() as tape:

            # Do this forward pass again with under the watchful eye of the GradientTape()
            features = self.backbone(data_aug)

            # Deal with the RPN: accumulate RoI, forward pass, loss calculation
            rpn_cls, rpn_bbox = self.rpnmodel.rpn(features, training=True)

            # I have no idea what on Earth is going on here,
            # for some reason this works but removing this
            # and starting the next line loss=... doesn't.
            loss = 0.0

            loss += tf.reduce_sum(
                tf.map_fn(
                    self.rpnmodel._compute_loss,
                    [rpn_cls, rpn_bbox, rpn_roi],
                    fn_output_signature=(tf.float32),
                )
            )

            # Now we're done with the big features map, so pool using
            # the RoI defined from above
            features_pool = tf.map_fn(
                self.roi_pool._pool_rois,
                (features, roi),
                fn_output_signature=tf.float32,
            )

            # Classifier forward pass accumulating loss
            cls, bbox = self.classmodel.classifier(features_pool, training=True)

            loss += tf.reduce_sum(
                tf.map_fn(
                    self.classmodel._compute_loss,
                    [cls, bbox, roi, labels],
                    fn_output_signature=(tf.float32),
                )
            )

        gradients = tape.gradient(
            loss,
            self.backbone.extractor.trainable_variables
            + self.rpnmodel.rpn.trainable_variables
            + self.classmodel.classifier.trainable_variables,
        )

        self.optimizer.apply_gradients(
            (grad, var)
            for grad, var in zip(
                gradients,
                self.backbone.extractor.trainable_variables
                + self.rpnmodel.rpn.trainable_variables
                + self.classmodel.classifier.trainable_variables,
            )
            if grad is not None
        )

        # Update metrics based on the labels and return loss
        self.compiled_metrics.update_state(
            data[1], self.classmodel.call((features_pool, roi))
        )
        return {"loss": loss, **{m.name: m.result() for m in self.metrics}}

    def test_step(self, data):

        """
        Run a model test step and return metrics and loss.

        Arguments:

        data : [tf.Tensor, int]
            Minibatch of data from the data loading classes.

        """

        # Run the feature extractor and decode labels
        features = self.backbone(data[0])
        labels = self.label_decoder(data[1])

        # Deal with the RPN: accumulate RoI, forward pass, loss calculation
        rpn_roi = tf.map_fn(self.rpnmodel._accumulate_roi, labels)
        rpn_cls, rpn_bbox = self.rpnmodel.rpn(features, training=False)
        loss = 0.0
        loss += tf.reduce_sum(
            tf.map_fn(
                self.rpnmodel._compute_loss,
                [rpn_cls, rpn_bbox, rpn_roi],
                fn_output_signature=(tf.float32),
            )
        )

        # Loop over images accumulating RoI proposals in forward mode
        features_pool, roi = self.roi_pool(
            (
                features,
                self.rpnmodel(features, input_images=False, output_images=False),
            )
        )

        # Classifier forward pass accumulating loss
        cls, bbox = self.classmodel.classifier(features_pool, training=False)
        loss += tf.reduce_sum(
            tf.map_fn(
                self.classmodel._compute_loss,
                [cls, bbox, roi, labels],
                fn_output_signature=(tf.float32),
            )
        )

        # Update metrics based on the labels and return loss
        self.compiled_metrics.update_state(
            data[1], self.classmodel.call((features_pool, roi))
        )
        return {"loss": loss, **{m.name: m.result() for m in self.metrics}}
