import tensorflow as tf
from models.AutoEncoder import AutoEncoder
from models.utils import reconstruction_distances


def baseline_ae_model(region_tensors, is_training, encoded_dims):
    squared_euclidean = squared_x = None
    for name in region_tensors:
        tensors = region_tensors[name]['tensors']
        filters = region_tensors[name]['filters']
        reuse = region_tensors[name]['reuse']
        scope = region_tensors[name]['scope']
        ae = AutoEncoder(tensors, encoded_dims=encoded_dims, filters=filters, is_training=is_training, reuse=reuse, name='compressor_{}'.format(scope))
        sq_x, sq_euclidean, cosine_similarity = reconstruction_distances(ae.input_tensor, ae.reconstruction)
        with tf.name_scope('latent_variables'):
            if squared_euclidean is None:
                squared_euclidean = sq_euclidean
                squared_x = sq_x
            else:
                squared_euclidean = squared_euclidean + sq_euclidean
                squared_x = squared_x + sq_x

        return tf.sqrt(squared_euclidean) / tf.sqrt(squared_x), ae.flatten_encoded, tf.reduce_mean(squared_euclidean)