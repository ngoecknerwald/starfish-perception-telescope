import tensorflow as tf


class LearningRateCallback(tf.keras.callbacks.Callback):
    def __init__(self, learning_rate, weight_decay):
        """

        Callback routine to set the learning rate and weight
        decay as a function of epoch number.

        learning_rate : dict
            {'epoch' : list of epoch number, 'rate' : list of learning rates}
        weight_decay : dict
            {'epoch' : list of epoch number, 'rate' : list of decay rates}

        """

        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    # Learning rate callback
    def lr_schedule(self, epoch):
        for _epoch, _value in zip(
            self.learning_rate["epochs"][::-1], self.learning_rate["values"][::-1]
        ):
            if _epoch <= epoch:
                return _value
        return self.learning_rate["values"][0]

    # Weight decay callback
    def wd_schedule(self, epoch):
        for _epoch, _value in zip(
            self.weight_decay["epochs"][::-1], self.weight_decay["values"][::-1]
        ):
            if _epoch <= epoch:
                return _value
        return self.weight_decay["values"][0]

    def on_epoch_begin(self, epoch, logs=None):

        _lr = self.lr_schedule(epoch)
        _wd = self.wd_schedule(epoch)

        tf.keras.backend.set_value(self.model.optimizer.learning_rate, _lr)
        tf.keras.backend.set_value(self.model.optimizer.weight_decay, _wd)

        print("\nEpoch %d: learning_rate=%.4f weight_decay=%.6f " % (epoch, _lr, _wd))
