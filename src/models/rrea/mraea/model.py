from __future__ import absolute_import

import keras
from keras.layers import *
from .layer import TR_GraphAttention
from keras import activations, constraints, initializers, regularizers
from keras import backend as K
from keras.layers import Layer, Dropout, LeakyReLU
import tensorflow as tf
import numpy as np


class TokenEmbedding(keras.layers.Embedding):
    """Embedding layer with weights returned."""

    def compute_output_shape(self, input_shape):
        return self.input_dim, self.output_dim

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs):
        return K.identity(self.embeddings)


def get_mraea_model(node_size, rel_size, node_hidden, rel_hidden, triple_size, batch_size, n_attn_heads=2,
                    dropout_rate=0,
                    gamma=3, lr=0.005, depth=2):
    adj_input = Input(shape=(None, 2))
    index_input = Input(shape=(None, 2), dtype='int64')
    val_input = Input(shape=(None,))
    rel_adj = Input(shape=(None, 2))
    ent_adj = Input(shape=(None, 2))

    org_feature = TokenEmbedding(node_size, node_hidden, trainable=True)(val_input)
    rel_feature = TokenEmbedding(rel_size, rel_hidden, trainable=True)(val_input)

    gat_in = [org_feature, rel_feature, adj_input, index_input, val_input, rel_adj, ent_adj]

    ent_feature, _, _ = TR_GraphAttention(node_size, activation='relu',
                                          rel_size=rel_size,
                                          depth=depth,
                                          attn_heads=n_attn_heads,
                                          triple_size=triple_size,
                                          attn_heads_reduction='average',
                                          dropout_rate=dropout_rate)(gat_in)
    ent_feature = Dropout(dropout_rate)(ent_feature)

    alignment_input = Input(shape=(None, 4))
    find = Lambda(lambda x: K.gather(reference=x[0], indices=K.cast(K.squeeze(x[1], axis=0), 'int32')))(
        [ent_feature, alignment_input])

    def loss_function(tensor):
        def dis(ll, rr):
            return K.sum(K.abs(ll - rr), axis=-1, keepdims=True)

        l, r, fl, fr = [tensor[:, 0, :], tensor[:, 1, :], tensor[:, 2, :], tensor[:, 3, :]]
        loss = K.relu(gamma + dis(l, r) - dis(l, fr)) + K.relu(gamma + dis(l, r) - dis(fl, r))
        return tf.reduce_sum(loss, keep_dims=True) / (batch_size)

    loss = Lambda(loss_function)(find)

    inputs = [adj_input, index_input, val_input, rel_adj, ent_adj]
    train_model = keras.Model(inputs=inputs + [alignment_input], outputs=loss)
    train_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=keras.optimizers.RMSprop(lr=lr))

    feature_model = keras.Model(inputs=inputs, outputs=[ent_feature])
    return train_model, feature_model
