import tensorflow as tf
from models.utils import base_dense_layer


class Estimator:
    def __init__(self, k, z, is_training=True):
        with tf.variable_scope('estimator'):
            self.input_tensor = z
            net = base_dense_layer(self.input_tensor, 32, name='dense_0', is_training=is_training)
            # net = base_dense(net, 256, name='dense_1', is_training=is_training)
            # net = base_dense(net, 128, name='dense_2', is_training=is_training)
            net = base_dense_layer(net, 64, name='dense_3', is_training=is_training)
            net = base_dense_layer(net, 32, name='dense_4', is_training=is_training)
            net = base_dense_layer(net, k, name='dense_5', is_training=is_training, bn=False)
            self.output_tensor = tf.nn.softmax(net, name='predicted_memebership')

