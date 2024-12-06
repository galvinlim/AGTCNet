from typing import Optional

import tensorflow as tf

from .constraints import MinMaxValue

class PositionalEncoding(tf.keras.layers.Layer):
    # Modified version of tfm.vision.layers.PositionalEncoding
    def __init__(self,
                 trainable_scale = True,
                 scale_initializer = tf.keras.initializers.Zeros(),
                 scale_constraint = MinMaxValue(min_value=0.0, max_value=1.0),
                 **kwargs):
        
        super(PositionalEncoding, self).__init__(**kwargs)
        self.trainable_scale = trainable_scale
        self.scale_initializer = scale_initializer
        self.scale_constraint = scale_constraint

    def _positional_encoding(self, max_pos, embedding_dim, dtype='float32'):

        positions = tf.range(max_pos)
        positions = tf.cast(positions, dtype)[:, tf.newaxis] # (T, 1)
        embedding_idx = tf.range(embedding_dim)[tf.newaxis, :] # (1, F)

        power = tf.cast(2 * (embedding_idx // 2), dtype)
        power /= tf.cast(embedding_dim, dtype)
        angles = 1. / tf.math.pow(10_000., power) # (1, F)
        radians = positions * angles # (T, F)

        sin = tf.math.sin(radians[:, 0::2]) # (T, F//2)
        cos = tf.math.cos(radians[:, 1::2]) # (T, F//2)
        pos_encoding = tf.concat([sin, cos], axis=-1) # (T, F)

        return pos_encoding

    def build(self, input_shape):
        # (..., T, F)
        frames = input_shape[-2]
        channels = input_shape[-1]

        pos_encoding = self._positional_encoding(frames, channels, dtype=self.dtype) # [frames_T, channels_F]
        self._pos_encoding = tf.expand_dims(pos_encoding, axis=0) # [1, frames_T, channels_F]

        if self.trainable_scale:
            self._rezero = Scale(initializer=self.scale_initializer, constraint=self.scale_constraint, name='rezero') # Scalar

        super(PositionalEncoding, self).build(input_shape)

    def call(self, inputs):

        pos_encoding = tf.cast(self._pos_encoding, inputs.dtype)
        
        if self.trainable_scale:
            pos_encoding = self._rezero(pos_encoding)
        
        outputs = inputs + pos_encoding

        return outputs

    def get_config(self):
        """Returns a dictionary containing the config used for initialization."""
        config = {'scale_initializer': self.scale_initializer,
                  'scale_constraint': self.scale_constraint}
        base_config = super(PositionalEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
  
class Scale(tf.keras.layers.Layer):
    def __init__(self,
                initializer: tf.keras.initializers.Initializer = 'ones',
                regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
                constraint: Optional[tf.keras.constraints.Constraint] = None,
                **kwargs):
        super(Scale, self).__init__(**kwargs)

        self.initializer = initializer
        self.regularizer = regularizer
        self.constraint = constraint

        self.scale = self.add_weight(name='scale',
                                      shape=[],
                                      dtype=self.dtype,
                                      initializer=self.initializer,
                                      regularizer=self.regularizer,
                                      constraint=self.constraint,
                                      trainable=True)

    def call(self, inputs):
        scale = tf.cast(self.scale, inputs.dtype)
        return scale * inputs

    def get_config(self):
        config = {'initializer': self.initializer,
                  'regularizer': self.regularizer,
                  'constraint': self.constraint}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))