import tensorflow as tf


def base_conv_layer(input_tensor,
                    filters,
                    strides=(2, 2),
                    kernels=(3, 3),
                    name='conv',
                    downsample=True,
                    is_training=True,
                    bn=True,
                    activation_fn=tf.nn.leaky_relu):
    with tf.variable_scope(name):
        if downsample:
            net = tf.layers.conv2d(input_tensor,
                                   filters,
                                   kernel_size=kernels,
                                   strides=strides,
                                   padding='SAME',
                                   name='conv',
                                   activation=None)
        else:
            net = tf.layers.conv2d_transpose(input_tensor,
                                             filters,
                                             kernel_size=kernels,
                                             strides=strides,
                                             padding='SAME',
                                             name='conv_transposed',
                                             activation=None)
        if bn:
            net = tf.layers.batch_normalization(net, training=is_training, name='batch_normalization')

        if activation_fn is not None:
            net = activation_fn(net)

    return net


def base_dense_layer(input_layer, units, name='dense', is_training=True, bn=True, activation_fn=tf.nn.leaky_relu):
    with tf.variable_scope(name):
        net = tf.layers.dense(input_layer,
                              units)
        if bn:
            net = tf.layers.batch_normalization(net, training=is_training, name='batch_normalization')

        if activation_fn is not None:
            net = activation_fn(net)

        return net


def count_trainable_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('number of parameters: ', total_parameters)


def reconstruction_distances(input_tensor, reconstruction):
    with tf.variable_scope('reconstruction_distances'):
        squared_x = tf.reduce_sum(tf.square(input_tensor),
                                  axis=[1, 2, 3],
                                  name='squared_x') + 1e-12
        squared_euclidean = tf.reduce_sum(tf.square(input_tensor - reconstruction),
                                          axis=[1, 2, 3],
                                          name='squared_euclidean') + 1e-12

        n1 = tf.nn.l2_normalize(input_tensor, [1, 2, 3])
        n2 = tf.nn.l2_normalize(reconstruction, [1, 2, 3])
        cosine_similarity = tf.reduce_sum(tf.multiply(n1, n2), axis=[1, 2, 3], name='cosine_similarity')
        return squared_x, squared_euclidean, cosine_similarity
