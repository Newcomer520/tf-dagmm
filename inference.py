import os
import glob
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from config import PIN_OBJECT, MAIN_OBJECT
from dagmm.dagmm import dagmm


def model_for_inference(encoded_dims=2, mixtures=5, use_pins=False):
    regions = PIN_OBJECT if use_pins else MAIN_OBJECT
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

    energy_tensors, z = dagmm(region_tensors, is_training=False, encoded_dims=encoded_dims, mixtures=mixtures)

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
        for idx, image_file in enumerate(image_files):
            image = cv2.imread(image_file)
            b, g, r = cv2.split(image)
            rgb_img = cv2.merge([r, g, b])
            rgb_img = rgb_img.astype(np.float32) / 255.0
            for region_name in region_tensors:
                tensors = region_tensors[region_name]['tensors']
                w = region_tensors[region_name]['width']
                h = region_tensors[region_name]['height']
                region = region_tensors[region_name]['region']
                if region == 'all':
                    sub_image = rgb_img
                else:
                    xmin, ymin, xmax, ymax = region
                    sub_image = rgb_img[ymin:ymax, xmin:xmax, :]
                sub_image = cv2.resize(sub_image, (h, w))
                current_batch[tensors].append(sub_image)

            if idx % batch_size == batch_size - 1 or idx == len(image_files) - 1:
                es, zs = sess.run([energy_tensors, z], feed_dict=current_batch)
                for region_name in region_tensors:
                    tensors = region_tensors[region_name]['tensors']
                    current_batch[tensors] = []
                for z_idx in range(z.get_shape()[1]):
                    results['z_{}'.format(z_idx)].extend(zs[:, z_idx])
                energies.extend(es)

    return pd.DataFrame(results).reindex(columns=columns).sort_values(by='energy')