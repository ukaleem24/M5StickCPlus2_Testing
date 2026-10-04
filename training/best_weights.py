import numpy as np
import tensorflow as tf


class RestoreBestWeights(tf.keras.callbacks.Callback):
    """Restores the best-`monitor` epoch's weights when training ends, without ever
    stopping training early.

    Needed because in the Keras version pinned here (2.10, required by
    tensorflow-directml-plugin), EarlyStopping(restore_best_weights=True) only
    restores when it actually triggers an early stop -- its on_train_end doesn't
    restore. With patience >= epochs it never triggers, so it silently keeps the
    last epoch's weights instead.

    The last epoch's weights are kept in `last_weights` so callers can compare.
    """

    def __init__(self, monitor="val_accuracy"):
        super().__init__()
        self.monitor = monitor
        self.best = -np.inf
        self.best_epoch = None
        self.best_weights = None
        self.last_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self.monitor)
        if current is not None and current > self.best:
            self.best = current
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        self.last_weights = self.model.get_weights()
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print(f"Restored weights from best epoch {self.best_epoch + 1} ({self.monitor}={self.best:.4f})")
