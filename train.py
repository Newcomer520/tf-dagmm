"""
DEEP AUTOENCODING GAUSSIAN MIXTURE MODEL FOR UNSUPERVISED ANOMALY DETECTION
paper: https://openreview.net/forum?id=BJJLHbb0-
"""

import tensorflow as tf
import os
from config import PIN_OBJECT, MAIN_OBJECT
from dagmm.dagmm import dagmm
from dagmm.utils import count_trainable_parameters
import re
import glob
from functools import partial
from argparse import ArgumentParser


def train(args):
    # reference for training/validation:
    #
    # input_tensor: [None, 64, 64, 3]
    best_folder = os.path.join(args.logdir, 'best')
    best_run = -1
    best_loss = 1e12
    os.makedirs(best_folder, exist_ok=True)
    last_checkpoint = tf.train.latest_checkpoint(args.logdir)
    if last_checkpoint is not None:
        global_step = int(re.search('-([0-9]+)', last_checkpoint).groups()[0]) + 1
    else:
        global_step = 1

    regions = PIN_OBJECT if args.use_pins else MAIN_OBJECT

    batch_tensors, handle, training_iterator, validation_iterator = make_dataset(regions, batch_size=args.batch_size)
    is_training_placeholder = tf.placeholder_with_default(tf.constant(True), [])

    region_tensors = {}
    for region_name in regions:
        images = batch_tensors[region_name]
        filters = regions[region_name]['filters']
        scope = regions[region_name]['scope']
        reuse = regions[region_name]['reuse']
        region_tensors[region_name] = {'tensors': images, 'filters': filters, 'reuse': reuse, 'scope': scope}

    *rest, loss, loss_reconstruction, es_mean, loss_sigmas_diag = dagmm(region_tensors, is_training_placeholder, encoded_dims=args.encoded_dims, mixtures=args.mixtures)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('loss_reconstruction', loss_reconstruction)
    tf.summary.scalar('energy_sample', es_mean)
    tf.summary.scalar('loss_sigmas_diag', loss_sigmas_diag)
    summary_op = tf.summary.merge_all()
    count_trainable_parameters()

    checkpoint_saver = tf.train.Saver(max_to_keep=100)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if last_checkpoint is not None:
            print('Restoring from checkpoint: {}'.format(last_checkpoint))
            checkpoint_saver.restore(sess, last_checkpoint)
        else:
            print('Training models from nothing')

        train_writer = tf.summary.FileWriter('{}/{}/train'.format(args.logdir, global_step), sess.graph)
        validate_writer = tf.summary.FileWriter('{}/{}/validate'.format(args.logdir, global_step), sess.graph)

        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())
        current_step = 0

        for epoch_idx in range(args.epoch):
            sess.run(training_iterator.initializer)
            while True:
                # training loop
                try:
                    _, l, les, lr, lsd, summ_train = sess.run([train_op, loss, es_mean, loss_reconstruction, loss_sigmas_diag, summary_op],
                                                              feed_dict={handle: training_handle, is_training_placeholder: True})
                    if l < 100 and 0 < l < best_loss:
                        print('best: checkpoint-{}-{}'.format(l, global_step + epoch_idx))
                        previous_best = glob.glob(os.path.join(best_folder, 'checkpoint-{}-{}.*'.format(best_loss, best_run)))
                        for f in previous_best:
                            os.remove(os.path.join(best_folder, f))
                        best_loss = l
                        best_run = global_step + epoch_idx
                        checkpoint_saver.save(sess, os.path.join(
                            best_folder, 'checkpoint-{}'.format(best_loss)), global_step=best_run)
                    current_step += 1
                except tf.errors.OutOfRangeError:
                    break
                except tf.errors._impl.InvalidArgumentError:
                    """
                    With the higher dimension of encoded vector, it's easy to get the issue of "uninvertable matrix".
                    Currently I could not solve this problem 
                    """
                    print('******** WARNING ********** gmm singularity occurred!')
                    last_checkpoint = tf.train.latest_checkpoint(args.logdir)
                    print('Restoring from the best latest checkpoint {}'.format(last_checkpoint))
                    checkpoint_saver.restore(sess, last_checkpoint)

            val_loss = 0
            val_cnt = 0
            sess.run(validation_iterator.initializer)
            while True:
                # validation loop
                try:
                    vl, summ_val = sess.run([loss, summary_op], feed_dict={handle: validation_handle, is_training_placeholder: False})
                    val_loss += vl
                    val_cnt += 1
                except tf.errors.OutOfRangeError:
                    break

            if (global_step + epoch_idx) % 10 == 0:
                train_writer.add_summary(summ_train, global_step + epoch_idx)
                validate_writer.add_summary(summ_val, global_step + epoch_idx)

            if (global_step + epoch_idx) % 100 == 0:
                print('checkpoint-{} saved'.format(global_step + epoch_idx))
                checkpoint_saver.save(sess, os.path.join(
                    args.logdir, 'checkpoint'), global_step=global_step + epoch_idx)

            print('{} current_epoch: {}, {}, {}, {}, val loss: {}'.format(global_step + epoch_idx, lr, les, lsd, l, val_loss / val_cnt))


def parse_function(filename, regions):
    """
    :param filename:
    :param regions:
    :param is_training:
    :return:
    """

    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_decoded = tf.cast(image_decoded, tf.float32) / 255.0
    output = {}

    for region in regions:
        w = regions[region]['width']
        h = regions[region]['height']
        if regions[region]['region'] != 'all':
            xmin, ymin, xmax, ymax = regions[region]['region']
            region_image = image_decoded[ymin:ymax, xmin:xmax, :]
        else:
            region_image = image_decoded
        region_image = tf.image.resize_images(region_image, (h, w))
        output[region] = region_image

    return output


def get_iterator(regions, folder, batch_size=32, buffer_size=200, num_parallel_calls=2):
    files = glob.glob(os.path.join(folder, '*.jpg'))
    dataset = tf.data.Dataset.from_tensor_slices(files)
    parse_fn = partial(parse_function, regions=regions)
    dataset = dataset.map(parse_fn, num_parallel_calls=num_parallel_calls).shuffle(buffer_size).batch(batch_size)
    return dataset.make_initializable_iterator(), dataset


def make_dataset(regions,
                 train_folder='/mnt/storage/ipython/dataset/P8_SMT/J0602-J0603/train/OK/',
                 validation_folder='/mnt/storage/ipython/dataset/P8_SMT/J0602-J0603/test/OK/',
                 batch_size=24,
                 buffer_size=1000):
    training_iterator, training_dataset = get_iterator(regions, train_folder, batch_size, buffer_size=buffer_size)
    validation_iterator, validation_dataset = get_iterator(regions, validation_folder, 1000, buffer_size=buffer_size)
    handle = tf.placeholder(tf.string)
    batch_tensors = tf.data.Iterator.from_string_handle(handle, output_types=training_dataset.output_types, output_shapes=training_dataset.output_shapes).get_next()
    return batch_tensors, handle, training_iterator, validation_iterator


if __name__ == '__main__':
    parser = ArgumentParser(description='Train a dagmm.')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--epoch', default=50000, type=int)
    parser.add_argument('--encoded_dims', default=2, type=int)
    parser.add_argument('--mixtures', default=6, type=int)
    parser.add_argument('--logdir', default='/home/i-lun/works/smt/tmp2', type=str)
    parser.add_argument('--batch_size', default=38, type=int)
    parser.add_argument('--decay_start', default=1000, type=int)
    parser.add_argument('--use_pins', dest='use_pins', action='store_true')
    parser.set_defaults(use_pins=False)
    args = parser.parse_args()

    train(args)
