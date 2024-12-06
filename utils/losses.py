import tensorflow as tf
import numpy as np

class Loge(tf.keras.losses.Loss):
    def __init__(self, bias=1e-10, eps=1 - np.log(2.0)):
        super().__init__()
        
        self.bias = bias
        self.eps = eps.astype(np.float32)

    def call(self, y_true, y_pred):
        # Ensure labels have the same shape
        y_true = tf.squeeze(y_true)

        # Compute cross-entropy loss
        cross_entropy_loss = - tf.reduce_sum(y_true * tf.math.log(y_pred + self.bias), axis=-1)
        
        # Apply the additional operations
        loss = tf.math.log(cross_entropy_loss + self.eps) - tf.math.log(self.eps)
        # tf loss class automatically reduce_mean()

        return loss