import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import constraints, initializers, regularizers

from tensorflow.keras.layers import Add

from .constraints import MinMaxValue

class Reshape(Layer):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape

    def call(self, inputs):
        return tf.reshape(inputs, self.target_shape)
    
class UnscaledDropout(Layer):
    def __init__(self, drop_rate, **kwargs):
        super(UnscaledDropout, self).__init__(**kwargs)
        self.drop_rate = drop_rate

    def call(self, inputs, training=None):
        if training:
            # Generate a random mask where values are either 0 or 1 based on drop_rate
            mask = tf.random.uniform(shape=tf.shape(inputs), minval=0, maxval=1) >= self.drop_rate
            # Multiply inputs by the mask
            output = inputs * tf.cast(mask, dtype=tf.float32)
            return output
        else:
            return inputs

class WeightedAdd(Layer):
    def __init__(self, 
                 weight_initializer='ones',
                 weight_regularizer=None,
                 weight_constraint=MinMaxValue(min_value=0.0, max_value=1.0),
                 **kwargs):
        super(WeightedAdd, self).__init__(**kwargs)

        self.weight_initializer = initializers.get(weight_initializer)
        self.weight_regularizer = regularizers.get(weight_regularizer)
        self.weight_constraint = constraints.get(weight_constraint)
        
    def build(self, inputs):
        # inputs: list
        input_size = len(inputs)
        self.weight = self.add_weight(shape=(input_size),
                                      dtype=tf.float32,
                                      initializer=self.weight_initializer,
                                      regularizer=self.weight_regularizer,
                                      constraint=self.weight_constraint,
                                      trainable=True,
                                         name="weight")
        
    def call(self, input):

        return Add()([arr * self.weight[i] for i, arr in enumerate(input)])