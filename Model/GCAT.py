import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout, LeakyReLU

from spektral.layers import ops
from spektral.layers.convolutional.conv import Conv
from spektral.layers.ops import modes

class GCAT(Conv):
    # Modified version of spektral.layers.GATConv
    def __init__(self,
        channels,
        mod_weight, # Layer(units/filters=channels*attn_heads)
        mod_val, # Layer(units/filters=channels*attn_heads)
        mod_attn_src, # [Layer(units/filters=1) for _ in range(attn_heads)]
        mod_attn_dst, # [Layer(units/filters=1) for _ in range(attn_heads)]
        attn_heads=1,
        concat_heads=True, # True: Concat | False: Mean
        add_self_loops=True, # Bool
        attn_actvn=LeakyReLU(alpha=0.2),
        attn_dropout_rate=0.0,
        activation=None,
        use_bias=True,
        return_attn_coef=False,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            activation=activation,
            use_bias=use_bias,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            bias_constraint=bias_constraint,
            **kwargs,
        )

        self.channels = channels
        self.weight_src_dst = mod_weight
        self.mod_val = mod_val
        self.attn_src = mod_attn_src 
        self.attn_dst = mod_attn_dst
        self.attn_heads = attn_heads
        self.concat_heads = concat_heads
        self.add_self_loops = add_self_loops
        self.attn_actvn = attn_actvn
        self.attn_dropout_rate = attn_dropout_rate
        self.return_attn_coef = return_attn_coef

        if concat_heads:
            self.output_dim = self.channels * self.attn_heads
        else:
            self.output_dim = self.channels
        
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]

        if self.use_bias:
            self.bias = self.add_weight(shape=[self.output_dim],
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        name='bias')
            
        self.attn_dropout = Dropout(self.attn_dropout_rate, dtype=self.dtype)

        self.built = True

    def call(self, inputs):
        x, a = inputs # (B,T,N,F) (B,T,N,N)

        mode = ops.autodetect_mode(x, a)
        assert mode != modes.SINGLE
        if K.is_sparse(a):
            a = tf.sparse.to_dense(a)

        output, attn_coef = self._call_attn(x, a) # (N,H,F)

        if self.concat_heads: # NO reduce mean
            output = tf.reshape(output, (-1, *output.shape[1:-2], self.attn_heads*self.channels)) # (N,HF)
        else:
            output = tf.reduce_mean(output, axis=-2) # (N,F)
        
        # (N,[H]F)
        if self.use_bias:
            output += tf.reshape(self.bias, (*((1,) * len(output.shape[:-1])), self.output_dim))
            
        output = self.activation(output)

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output

    def _call_attn(self, feat, adj):
        adj_nodes_shape = tf.shape(adj)[:-1]
        if self.add_self_loops:
            adj = tf.linalg.set_diag(adj, tf.ones(adj_nodes_shape, adj.dtype))

        # (B,T,N,F)
        feat_src_dst = self.weight_src_dst(feat) # (N,H*F)
        feat_src_dst = tf.reshape(feat_src_dst, (-1, *feat_src_dst.shape[1:-1], self.attn_heads, self.channels)) # (N,H,F)
        feat_src = feat_dst = feat_src_dst
        
        feat_val = self.mod_val(feat)
        feat_val = tf.reshape(feat_val, (-1, *feat_val.shape[1:-1], self.attn_heads, self.channels)) # (N,H,F)
    
        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j            
        attn_src = tf.stack([self.attn_src[h](feat_src[...,h,:]) for h in range(self.attn_heads)], axis=-1) # (N,H,F): for H, (N,F) @ (F,1) = (N,1) -> stack(N,1,H) -> (S,1,H)
        attn_dst = tf.stack([self.attn_dst[h](feat_dst[...,h,:]) for h in range(self.attn_heads)], axis=-1) # (N,H,F): for H, (N,F) @ (F,1) = (N,1) -> stack(N,1,H) -> (D,1,H)

        # broadcast nodes
        attn_dst = tf.einsum("...NIH -> ...INH", attn_dst) # (D,1,H) -> (1,D,H)
        attn_coef = attn_src + attn_dst # (S,1,H) + (1,D,H) = (S,D,H)
        attn_coef = self.attn_actvn(attn_coef)
            
        # compute edge_softmax
        adj_mask = tf.where(adj == 1.0, 0.0, -10e9) # (S,D)
        adj_mask = tf.cast(adj_mask, dtype=attn_coef.dtype)
        attn_coef += adj_mask[..., None] # (S,D,H) + mask(S,D,1) = (S,D,H)
        attn_coef_softmax = tf.nn.softmax(attn_coef, axis=-3) # (S,D,H) softmax @S-node | for D-node, sum(S-nodes)=1

        attn_coef_drop = self.attn_dropout(attn_coef_softmax)

        # message passing
        feat_out = tf.einsum("...SDH , ...SHF -> ...DHF", attn_coef_drop, feat_val)

        return feat_out, attn_coef_softmax

    @property
    def config(self):
        return {
            "channels": self.channels,
            "attn_heads": self.attn_heads,
            "concat_heads": self.concat_heads,
            "return_attn_coef": self.return_attn_coef,
        }