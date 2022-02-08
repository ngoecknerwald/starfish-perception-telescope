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

        super(CustomLearningRateScheduler, self).__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    # Learning rate callback
    def lr_schedule(epoch):
        for i, _epoch in self.learning_rate["epochs"][::-1]:
            if _epoch <= epoch:
                return self.learning_rate["values"][i]
        return self.learning_rate["values"][0]

    # Weight decay callback
    def wd_schedule(epoch):
        for i, _epoch in self.weight_decay["epochs"][::-1]:
            if _epoch <= epoch:
                return self.weight_decay["values"][i]
        return self.weight_decay["values"][0]

    def on_epoch_begin(self, epoch, logs=None):

        _lr = self.lr_schedule(epoch)
        _wd = self.wd_schedule(epoch)

        tf.keras.backend.set_value(self.model.optimizer.learning_rate, _lr)
        tf.keras.backend.set_value(self.model.optimizer.weight_decay, _wd)

        print("\nEpoch %d: learning_rate=%.4f weight_decay=%.6f " % (epoch, _lr, _wd))
