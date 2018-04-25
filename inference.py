import os
import glob
import tensorflow as tf
import pandas as pd
from config import get_region
from models.dagmm import dagmm
from models.baseline_ae import baseline_ae_model
import datetime
from utils import feed_image_to_tensors


def model_for_inference(pattern_name, encoded_dims=2, mixtures=5, use_pins=False, latent_dims=2, baseline=False):
    regions = get_region(pattern_name)
    region_tensors = {}
    for region_name in regions:
        w = regions[region_name]['width']
        h = regions[region_name]['height']
        filters = regions[region_name]['filters']
        scope = regions[region_name]['scope']
        reuse = regions[region_name]['reuse']
        region = regions[region_name]['region']
        tensors = tf.placeholder(tf.float32, (None, h, w, 3))
        region_tensors[region_name] = {'filters': filters, 'tensors': tensors, 'reuse': reuse, 'width': w, 'height': h, 'region': region, 'scope': scope}

    if baseline:
        print('use baseline model')
        rel_dist, z, squared_dist = baseline_ae_model(region_tensors, is_training=tf.constant(False, name='is_training'), encoded_dims=encoded_dims)
        return region_tensors, rel_dist, tf.concat([z, tf.reshape(rel_dist,(-1, 1))], axis=1)
    else:
        energy_tensors, z, *rest = dagmm(region_tensors, is_training=tf.constant(False), encoded_dims=encoded_dims, mixtures=mixtures, latent_dims=latent_dims)
        return region_tensors, energy_tensors, z


def inference(folder,
              region_tensors,
              energy_tensors,
              z,
              checkpoint='/home/i-lun/works/smt/tmp/checkpoint-8400',
              batch_size=24):
    image_files = glob.glob(os.path.join(folder, '*.jpg'))
    columns = ['file', 'energy']
    results = {'file': list(map(lambda f: os.path.basename(f), image_files)), 'energy': []}
    energies = results['energy']

    for z_idx in range(z.get_shape()[1]):
        results['z_{}'.format(z_idx)] = []
        columns.append('z_{}'.format(z_idx))

    checkpoint_saver = tf.train.Saver()

    with tf.Session() as sess:
        checkpoint_saver.restore(sess, checkpoint)
        current_batch = {}
        for region_name in region_tensors:
            tensors = region_tensors[region_name]['tensors']
            current_batch[tensors] = []

        starttime = datetime.datetime.now()
        for idx, image_file in enumerate(image_files):
            feed_image_to_tensors(image_file, region_tensors, current_batch)

            if idx % batch_size == batch_size - 1 or idx == len(image_files) - 1:
                es, zs = sess.run([energy_tensors, z], feed_dict=current_batch)
                for region_name in region_tensors:
                    tensors = region_tensors[region_name]['tensors']
                    current_batch[tensors] = []
                for z_idx in range(z.get_shape()[1]):
                    results['z_{}'.format(z_idx)].extend(zs[:, z_idx])
                energies.extend(es)

        endtime = datetime.datetime.now()
        t = endtime - starttime
        print('required inference time for {} images: {} sec'.format(len(image_files), t))
    return pd.DataFrame(results).reindex(columns=columns).sort_values(by='energy')
