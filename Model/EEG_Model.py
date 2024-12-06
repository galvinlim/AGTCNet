from tensorflow.keras.layers import Flatten, Activation, Add, Concatenate
from tensorflow.keras.layers import Conv1D, SeparableConv1D
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import DepthwiseConv2D, SeparableConv2D
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import PReLU
from tensorflow.keras import Sequential
from tensorflow.keras import initializers
from tensorflow.keras.constraints import MaxNorm

from utils.wrapper import ModelWrapper, GConvWrapper
from .GCAT import GCAT
from .positional_encoding import PositionalEncoding
from .layers import Reshape
from .constraints import MinMaxValue

class AGTCNet(ModelWrapper):
   def __init__(self, dataset):
      super().__init__(dataset)

      # Channel-wise Temporal Convolution (CTC) Module
      # (S, N, T, F) -> (SN, T, F)
      self.ctconv = Sequential([Conv1D(8, 32, strides=1, padding='valid', use_bias=False),
                                BatchNormalization(axis=-1)],
                               name='ctconv')

      self.ctpool = AveragePooling1D(4, strides=2)

      # Graph Convolutional Attention Network (GCAT) Module
      # (S, N, T, F) (S, N, N) -> (S, T, N, F) (S, T, N, N)
      # Layer(units/filters=channels*attn_heads)
      GCAT_mod_weight = Sequential([SeparableConv2D(16*2, (8,1), depth_multiplier=4, strides=(1,1), padding='same', use_bias=False),
                                    BatchNormalization(axis=-1),
                                    Activation('selu')])
      GCAT_mod_val = Sequential([SeparableConv2D(16*2, (8,1), depth_multiplier=4, strides=(1,1), padding='same', use_bias=False),
                                 BatchNormalization(axis=-1),
                                 Activation('selu')])
      
      # [Layer(units/filters=1) for _ in range(attn_heads)]
      GCAT_mod_attn_src = [Sequential([SeparableConv2D(1, (2,1), depth_multiplier=4, strides=(1,1), padding='same', use_bias=False),
                                       BatchNormalization(axis=-1),
                                       Activation('selu')]) for _ in range(2)]
      GCAT_mod_attn_dst = [Sequential([SeparableConv2D(1, (2,1), depth_multiplier=4, strides=(1,1), padding='same', use_bias=False),
                                       BatchNormalization(axis=-1),
                                       Activation('selu')]) for _ in range(2)]
      
      self.gcat = GConvWrapper(GCAT(channels=16, 
                                    mod_weight=GCAT_mod_weight,
                                    mod_val=GCAT_mod_val,
                                    mod_attn_src=GCAT_mod_attn_src,
                                    mod_attn_dst=GCAT_mod_attn_dst,
                                    attn_heads=2,
                                    concat_heads=False,
                                    add_self_loops=True,
                                    attn_actvn=PReLU(shared_axes=[1]),
                                    attn_dropout_rate=0.2,
                                    activation=None,
                                    use_bias=True))

      self.gcat_norm = Sequential([BatchNormalization(axis=-1) ,
                                   Activation('selu'),
                                   Dropout(0.25)],
                                  name='gcat_norm')

      # Global Convolutional Adaptive Pooling (GCAP) Module
      # (S, N, T, F) -> (S, T, F)
      self.chpool = Sequential([DepthwiseConv2D((self.N, 1), depth_multiplier=2, strides=(1,1), padding='valid', use_bias=False, depthwise_constraint=MaxNorm(1., axis=0)),
                                BatchNormalization(axis=-1),
                                Activation('selu')],
                               name='chpool')

      # Global Temporal Convolution (GTC) Module
      # (S, T, F)
      self.tpool1 = Sequential([AveragePooling1D(4, strides=4),
                                Dropout(0.25)],
                               name='tpool1')
      
      self.tconv = Sequential([SeparableConv1D(96, 8, depth_multiplier=4, strides=1, padding='same', use_bias=False),
                               BatchNormalization(axis=-1),
                               Activation('selu')],
                              name='tconv')

      self.tpool2 = Sequential([AveragePooling1D(4, strides=4),
                                Dropout(0.25)],
                               name='tpool2')

      # Temporal Context Enhancement (TCE) Module
      # (S, T, F)
      self.tce_pos_encoder = PositionalEncoding(trainable_scale=True,
                                                scale_initializer=initializers.Zeros(),
                                                scale_constraint=MinMaxValue(min_value=0.0, max_value=1.0))

      self.tce_mha = MultiHeadAttention(num_heads=2, key_dim=8, value_dim=8, dropout=0.6, use_bias=True, attention_axes=None)
      self.tce_mha_dropout = Sequential([BatchNormalization(axis=-1),
                                         Dropout(0.3)])

      self.tce_conv = Sequential([SeparableConv1D(96, 2, depth_multiplier=4, strides=1, padding='same', use_bias=False),
                                  BatchNormalization(axis=-1),
                                  Activation('selu'),
                                  Dropout(0.2)],
                                 name='tce_conv')

      # Classification Module
      # (S, T, F) -> (S, TF)
      self.classifier = Sequential([Dense(self.num_classes, use_bias=True, kernel_constraint=MaxNorm(0.25, axis=0)),
                                    Activation('softmax', name='softmax')],
                                    name='classifier')

   def call(self, batch):
      feats, graphs = batch
      out = self.input_reshape(feats) # (S, N, TF) -> (S, N, T, F)

      # Channel-wise Temporal Convolution (CTC) Module
      # (S, N, T, F)
      out = Reshape((-1, *out.shape[-2:]))(out) # (SN, T, F)
      out = self.ctconv(out)
      out = self.ctpool(out)
      out = Reshape((-1, self.N, *out.shape[-2:]))(out) # (S, N, T, F)

      # Graph Convolutional Attention Network (GCAT) Module
      # (S, N, T, F) (S, N, N)
      out0 = out
      out = self.gcat([out, graphs])
      out = self.gcat_norm(out)
      out = Concatenate(axis=-1)([out0, out])

      # Global Convolutional Adaptive Pooling (GCAP) Module
      # (S, N, T, F) -> (S, T, F)
      out = self.chpool(out)
      out = out[..., 0, :, :]

      # Global Temporal Convolution (GTC) Module
      # (S, T, F)
      out = self.tpool1(out)
      out = self.tconv(out)
      out = self.tpool2(out)

      # Temporal Context Enhancement (TCE) Module
      # (S, T, F)
      out = self.tce_pos_encoder(out)

      out0 = out
      out = self.tce_mha(out, out)
      out = self.tce_mha_dropout(out)
      out = Add()([out0, out])

      out0 = out
      out = self.tce_conv(out)
      out = Add()([out0, out])

      # Classification Module
      # (S, T, F) -> (S, TF)
      out = Flatten()(out)
      out = self.classifier(out)

      return out