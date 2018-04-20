import tensorflow as tf
import numpy as np
from models.utils import base_conv_layer, base_dense_layer
from config import FILTERS


def encoder(input_tensor, filters, is_training=True):
    """
    :param input_tensor: need to be preprocessed into [0, 1]
    :param filters:
    :param is_training:
    :return:
    """
    with tf.variable_scope('encoder'):
        net = base_conv_layer(input_tensor, 16, (1, 1), name='conv_0', is_training=is_training)
        for idx, num_filter in enumerate(filters):
            name = 'conv_{}'.format(str(idx + 1)) if idx < len(filters) - 1 else 'encoded'
            net = base_conv_layer(net, num_filter, name=name, is_training=is_training)
    return net


def decoder(encoded_tensor, filters, is_training=True):
    """
    :param encoded_tensor:
    :param filters: filters used in the relative encoder, therefore they will be reversed in the decoder
    :param is_training:
    :return:
    """
    reversed_filters = np.flip(filters, axis=0)
    net = tf.reshape(encoded_tensor, [-1, 1, 1, reversed_filters[0]])

    with tf.variable_scope('decoder'):
        for idx, num_filter in enumerate(reversed_filters[1:]):
            net = base_conv_layer(net,
                                  num_filter,
                                  name='conv_trans_{}'.format(str(idx)),
                                  is_training=is_training,
                                  downsample=False)
        net = base_conv_layer(net,
                              16,
                              name='conv_trans_{}'.format(str(len(filters))),
                              is_training=is_training,
                              downsample=False)
        net = base_conv_layer(net,
                              3,
                              strides=(1, 1),
                              name='reconstruction',
                              is_training=is_training,
                              downsample=False,
                              bn=False,
                              activation_fn=tf.nn.sigmoid)
    return net


class AutoEncoder:
    def __init__(self, input_tensor, encoded_dims=8, filters=FILTERS, is_training=True, name='autoencoder', reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            self.input_tensor = input_tensor
            encoded_tensor = encoder(self.input_tensor, is_training=is_training, filters=filters)
            encoded_shape = encoded_tensor.get_shape().as_list()
            dims = np.prod(encoded_shape[1:])
            encoded_gap = tf.reduce_sum(encoded_tensor, axis=[1, 2])
            eg_dims = encoded_gap.get_shape()[1]
            encoded_gap = tf.reshape(encoded_gap, (-1, eg_dims))
            self.flatten_encoded = net = base_dense_layer(encoded_gap, encoded_dims, name='flatten_encoded', is_training=is_training, bn=False, activation_fn=None)
            net = base_dense_layer(net, eg_dims, is_training=is_training, name='flatten_restored_0')
            net = base_dense_layer(net, dims, is_training=is_training, name='flatten_restored_1')
            net = tf.reshape(net, (-1, encoded_shape[1], encoded_shape[2], encoded_shape[2]))
            self.reconstruction = decoder(net, filters=filters, is_training=is_training)
