import tensorflow as tf


kernel_initializer = tf.truncated_normal_initializer(stddev=0.02, dtype=tf.float32)


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
                                   kernel_initializer=kernel_initializer,
                                   name='conv',
                                   activation=None)
        else:
            net = tf.layers.conv2d_transpose(input_tensor,
                                             filters,
                                             kernel_size=kernels,
                                             strides=strides,
                                             padding='SAME',
                                             kernel_initializer=kernel_initializer,
                                             name='conv_transposed',
                                             activation=None)
        if bn:
            net = tf.layers.batch_normalization(net, training=is_training, name='batch_normalization')
        net = activation_fn(net)

    return net


def base_dense_layer(input_layer, units, name='dense', is_training=True, bn=True, activation_fn=tf.nn.leaky_relu):
    with tf.variable_scope(name):
        net = tf.layers.dense(input_layer,
                              units,
                              kernel_initializer=kernel_initializer)
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