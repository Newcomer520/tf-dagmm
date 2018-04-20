import tensorflow as tf
import numpy as np
from models.AutoEncoder import AutoEncoder
from models.Estimator import Estimator
from models.utils import base_dense_layer, reconstruction_distances
import math


def dagmm(region_tensors, is_training, encoded_dims=2, mixtures=3, lambda_1=0.1, lambda_2=0.005, use_cosine_similarity=False, latent_dims=2):
    """
    :param region_tensors: restore the related tensors
    :param is_training: a tensorflow placeholder to indicate whether it is in the training phase or not
    :param encoded_dims:
    :param mixtures:
    :param lambda_1:
    :param lambda_2:
    :param use_cosine_similarity:
    :param latent_dims: reduce the dimension of encoded vector to a smaller one
    :return:
    """

    squared_x = squared_euclidean = z = None
    for name in region_tensors:
        tensors = region_tensors[name]['tensors']
        filters = region_tensors[name]['filters']
        reuse = region_tensors[name]['reuse']
        scope = region_tensors[name]['scope']
        ae = AutoEncoder(tensors, encoded_dims=encoded_dims, filters=filters, is_training=is_training, reuse=reuse, name='compressor_{}'.format(scope))
        reduced_latent = base_dense_layer(ae.flatten_encoded, latent_dims, 'reducer_{}'.format(name), is_training=is_training, bn=False, activation_fn=None)
        sq_x, sq_euclidean, cosine_similarity = reconstruction_distances(ae.input_tensor, ae.reconstruction)
        with tf.name_scope('latent_variables'):
            if use_cosine_similarity:
                relative_euclidean = tf.sqrt(sq_euclidean) / tf.sqrt(sq_x)
                relative_euclidean = tf.reshape(relative_euclidean, [-1, 1])
                cosine_similarity = tf.reshape(cosine_similarity, [-1, 1])
                distances = tf.concat([relative_euclidean, cosine_similarity], axis=1)
            else:
                distances = tf.sqrt(sq_euclidean) / tf.sqrt(sq_x)
                distances = tf.reshape(distances, [-1, 1])

            if squared_x is None:
                squared_x = sq_x
                squared_euclidean = sq_euclidean
                z = tf.concat([reduced_latent, distances], axis=1)
            else:
                squared_x = squared_x + sq_x
                squared_euclidean = squared_euclidean + sq_euclidean
                z = tf.concat([z, reduced_latent, distances], axis=1)
    with tf.name_scope('n_count'):
        n_count = tf.shape(z)[0]
        n_count = tf.cast(n_count, tf.float32)

    estimator = Estimator(mixtures, z, is_training=is_training)
    gammas = estimator.output_tensor

    with tf.variable_scope('gmm_parameters'):
        phis = tf.get_variable('phis', shape=[mixtures], initializer=tf.ones_initializer(), dtype=tf.float32, trainable=False)
        mus = tf.get_variable('mus', shape=[mixtures, z.get_shape()[1]], initializer=tf.ones_initializer(), dtype=tf.float32, trainable=False)

        init_sigmas = 0.5 * np.expand_dims(np.identity(z.get_shape()[1]), axis=0)
        init_sigmas = np.tile(init_sigmas, [mixtures, 1, 1])
        init_sigmas = tf.constant_initializer(init_sigmas)
        sigmas = tf.get_variable('sigmas', shape=[mixtures, z.get_shape()[1], z.get_shape()[1]], initializer=init_sigmas, dtype=tf.float32, trainable=False)

        sums = tf.reduce_sum(gammas, axis=0)
        sums_exp_dims = tf.expand_dims(sums, axis=-1)

        phis_ = sums / n_count
        mus_ = tf.matmul(gammas, z, transpose_a=True) / sums_exp_dims

        def assign_training_phis_mus():
            with tf.control_dependencies([phis.assign(phis_), mus.assign(mus_)]):
                return [tf.identity(phis), tf.identity(mus)]

        phis, mus = tf.cond(is_training, assign_training_phis_mus, lambda: [phis, mus])

        phis_exp_dims = tf.expand_dims(phis, axis=0)
        phis_exp_dims = tf.expand_dims(phis_exp_dims, axis=-1)
        phis_exp_dims = tf.expand_dims(phis_exp_dims, axis=-1)

        zs_exp_dims = tf.expand_dims(z, 1)
        zs_exp_dims = tf.expand_dims(zs_exp_dims, -1)
        mus_exp_dims = tf.expand_dims(mus, 0)
        mus_exp_dims = tf.expand_dims(mus_exp_dims, -1)

        zs_minus_mus = zs_exp_dims - mus_exp_dims

        sigmas_ = tf.matmul(zs_minus_mus, zs_minus_mus, transpose_b=True)
        broadcast_gammas = tf.expand_dims(gammas, axis=-1)
        broadcast_gammas = tf.expand_dims(broadcast_gammas, axis=-1)
        sigmas_ = broadcast_gammas * sigmas_
        sigmas_ = tf.reduce_sum(sigmas_, axis=0)
        sigmas_ = sigmas_ / tf.expand_dims(sums_exp_dims, axis=-1)
        sigmas_ = add_noise(sigmas_)

        def assign_training_sigmas():
            with tf.control_dependencies([sigmas.assign(sigmas_)]):
                return tf.identity(sigmas)

        sigmas = tf.cond(is_training, assign_training_sigmas, lambda: sigmas)

    with tf.name_scope('loss'):
        loss_reconstruction = tf.reduce_mean(squared_euclidean, name='loss_reconstruction')
        inversed_sigmas = tf.expand_dims(tf.matrix_inverse(sigmas), axis=0)
        inversed_sigmas = tf.tile(inversed_sigmas, [tf.shape(zs_minus_mus)[0], 1, 1, 1])
        energy = tf.matmul(zs_minus_mus, inversed_sigmas, transpose_a=True)
        energy = tf.matmul(energy, zs_minus_mus)
        energy = tf.squeeze(phis_exp_dims * tf.exp(-0.5 * energy), axis=[2, 3])
        energy_divided_by = tf.expand_dims(tf.sqrt(2.0 * math.pi * tf.matrix_determinant(sigmas)), axis=0) + 1e-12
        energy = tf.reduce_sum(energy / energy_divided_by, axis=1) + 1e-12
        energy = -1.0 * tf.log(energy)
        energy_mean = tf.reduce_sum(energy) / n_count
        loss_sigmas_diag = 1.0 / tf.matrix_diag_part(sigmas)
        loss_sigmas_diag = tf.reduce_sum(loss_sigmas_diag)
        loss = loss_reconstruction + lambda_1 * energy_mean + lambda_2 * loss_sigmas_diag

    return energy, z, loss, loss_reconstruction, energy_mean, loss_sigmas_diag


def add_noise(mat, stdev=0.001):
    """
    :param mat: should be of shape(k, d, d)
    :param stdev: the standard deviation of noise
    :return: a matrix with little noises
    """
    with tf.name_scope('gaussian_noise'):
        dims = mat.get_shape().as_list()[1]
        noise = stdev + tf.random_normal([dims], 0, stdev * 1e-1)
        noise = tf.diag(noise)
        noise = tf.expand_dims(noise, axis=0)
        noise = tf.tile(noise, (mat.get_shape()[0], 1, 1))
    return mat + noise

