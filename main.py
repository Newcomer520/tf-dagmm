import os
import glob
import tensorflow as tf
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from inference import inference, model_for_inference
from functools import partial
from sklearn.manifold import TSNE
from utils import feed_image_to_tensors
import cv2


def summary_report(inference_fn,
                   train_folder='/mnt/storage/ipython/dataset/P8_SMT/J0602-J0603/train/OK/',
                   test_OK_folder='/mnt/storage/ipython/dataset/P8_SMT/J0602-J0603/test/OK/',
                   test_NG_folder='/mnt/storage/ipython/dataset/P8_SMT/J0602-J0603/test/NG/',
                   saved_in=None):
    train_results = inference_fn(train_folder)
    train_len = train_results.shape[0]
    energies = train_results['energy'].tolist()
    thresholds = [energies[int(train_len * 0.95)], energies[int(train_len * 0.97)], energies[int(train_len * 0.99)]]
    test_OK_results = inference_fn(test_OK_folder)
    test_OK_results['label'] = 0
    test_NG_results = inference_fn(test_NG_folder)
    test_NG_results['label'] = 1
    test_results = pd.concat([test_OK_results, test_NG_results], ignore_index=True)

    labels = test_results.label
    energies = test_results.energy
    average_precision = average_precision_score(labels, energies)
    precision, recall, th = precision_recall_curve(labels, energies, pos_label=1)
    false_positive = test_OK_results[test_OK_results.energy >= th[0]].shape[0]

    fig_train_scatter = plt.figure(1)
    fig_train_scatter.set_size_inches(25, 10)
    ax = fig_train_scatter.add_subplot(111, projection='3d')
    cax = ax.scatter(train_results.z_0, train_results.z_1, train_results.z_2, c=train_results.energy, cmap=plt.cm.get_cmap('jet'),
                     vmin=min(test_results.energy), vmax=max(test_results.energy))
    fig_train_scatter.colorbar(cax)
    plt.title('Sample Energy Colormap\n(Higher ones being anomaly more possibily)')
    ax.set_xlabel('z_0')
    ax.set_ylabel('z_1')
    ax.set_zlabel('z_2')

    fig_test_scatter = plt.figure(2)
    fig_test_scatter.set_size_inches(25, 10)
    ax = fig_test_scatter.add_subplot(111, projection='3d')
    ax.scatter(test_OK_results.z_0, test_OK_results.z_1, test_OK_results.z_2, c=test_OK_results.energy, cmap=plt.cm.get_cmap('jet'),
               vmin=min(test_results.energy), vmax=max(test_results.energy))
    cax = ax.scatter(test_NG_results.z_0, test_NG_results.z_1, test_NG_results.z_2, c=test_NG_results.energy, cmap=plt.cm.get_cmap('jet'), marker='X', s=80,
                     vmin=min(test_results.energy), vmax=max(test_results.energy))
    fig_test_scatter.colorbar(cax)
    plt.title('Sample Energy Colormap\n(Higher ones being anomaly more possibily)')
    ax.set_xlabel('z_0')
    ax.set_ylabel('z_1')
    ax.set_zlabel('z_2')
    print(precision)
    print(recall)
    print(th)

    fig_hist = plt.figure(3)
    ax = fig_hist.add_subplot(111)
    ax.hist(test_OK_results.energy, color='blue', label='OK')
    ax.hist(test_NG_results.energy, color='red', label='NG')
    ax.legend()

    fig_pr_curve = plt.figure(4)
    ax = fig_pr_curve.add_subplot(111)
    ax.step(recall, precision, color='b', alpha=0.2, where='post')
    ax.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    ax.plot([0, 1], [precision[0], precision[0]], '--', linewidth=1, color='red')
    ax.text(0.3, precision[0] - 0.05, '{:.2f}%, fp: {:0d}'.format(precision[0] * 100, false_positive), color='red')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('OK: {}, NG: {}'.format(test_OK_results.shape[0], test_NG_results.shape[0]))
    plt.suptitle('PR curve: AP = {:0.2f}'.format(average_precision))

    fig_tsne = plt.figure(5)
    ax = fig_tsne.add_subplot(111)
    test_results_array = test_results.ix[:, 'z_0':].as_matrix()
    tsne_scatters = TSNE(n_components=2).fit_transform(test_results_array[:, 0:-1])
    ax.scatter(tsne_scatters[len(test_OK_results):, 0], tsne_scatters[len(test_OK_results):, 1], c=test_NG_results.energy, cmap=plt.cm.get_cmap('jet'), marker='x',
               vmin=min(test_results.energy), vmax=max(test_results.energy))
    cax = ax.scatter(tsne_scatters[: len(test_OK_results), 0], tsne_scatters[: len(test_OK_results), 1], c=test_OK_results.energy, cmap=plt.cm.get_cmap('jet'),
                     vmin=min(test_results.energy), vmax=max(test_results.energy))
    plt.colorbar(cax)
    plt.title('TSNE result')

    if not saved_in:
        plt.show()
    else:
        os.makedirs(saved_in, exist_ok=True)
        train_results.to_csv(os.path.join(saved_in, 'train_results.csv'), index=False)
        test_results.to_csv(os.path.join(saved_in, 'test_results.csv'), index=False)
        fig_train_scatter.savefig(os.path.join(saved_in, 'train_scatter.png'))
        fig_test_scatter.savefig(os.path.join(saved_in, 'test_scatter.png'))
        fig_hist.savefig(os.path.join(saved_in, 'hist.png'))
        fig_pr_curve.savefig(os.path.join(saved_in, 'pr_curve.png'))
        fig_tsne.savefig(os.path.join(saved_in, 'tsne.png'))

    print('thresholds: 95%: {}, 97%: {}, 99%: {}'.format(thresholds[0], thresholds[1], thresholds[2]))


def save_reconstruction_images(checkpoint,
                               region_tensors,
                               saved_in,
                               test_OK_folder='/mnt/storage/ipython/dataset/P8_SMT/J0602-J0603/test/OK/',
                               test_NG_folder='/mnt/storage/ipython/dataset/P8_SMT/J0602-J0603/test/NG/',):
    checkpoint_saver = tf.train.Saver()

    saved_OK_folder = os.path.join(saved_in, 'OK')
    saved_NG_folder = os.path.join(saved_in, 'NG')
    os.makedirs(saved_OK_folder, exist_ok=True)
    os.makedirs(saved_NG_folder, exist_ok=True)
    handles = [
        {'image_files': glob.glob(os.path.join(test_OK_folder, '*.jpg')), 'saved_in': saved_OK_folder},
        {'image_files': glob.glob(os.path.join(test_NG_folder, '*.jpg')), 'saved_in': saved_NG_folder},
    ]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        checkpoint_saver.restore(sess, checkpoint)
        reconstruction = sess.graph.get_tensor_by_name('compressor_main/decoder/reconstruction/Sigmoid:0')

        for handle in handles:
            current_batch = {}
            raw_images = []
            for region_name in region_tensors:
                tensors = region_tensors[region_name]['tensors']
                current_batch[tensors] = []

            for image_file in handle['image_files']:
                raw_image = feed_image_to_tensors(image_file, region_tensors, current_batch)
                raw_images.append(raw_image)

            rec_images = sess.run([reconstruction], feed_dict=current_batch)
            for idx, raw_image in enumerate(raw_images):
                image_file = handle['image_files'][idx]
                basename = os.path.basename(image_file)[:-4]
                # raw_image = cv2.resize(raw_image, (128, 128))
                # raw_image = (raw_image * 255).astype(np.uint8)
                rec_image = (rec_images[0][idx] * 255).astype(np.uint8)
                fig, (ax1, ax2) = plt.subplots(1, 2)
                plt.suptitle(basename, fontsize=16)
                ax1.axis('off')
                ax2.axis('off')
                ax1.imshow(raw_image)
                ax2.imshow(rec_image)
                ax1.title.set_text('Raw')
                ax2.title.set_text('Reconstruction')
                plt.savefig(os.path.join(handle['saved_in'], '{}.png'.format(basename)))
                plt.close()


def main():
    parser = ArgumentParser(description='Train a models.')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--epoch', default=50000, type=int)
    parser.add_argument('--encoded_dims', default=160, type=int)
    parser.add_argument('--latent_dims', default=6, type=int)
    parser.add_argument('--mixtures', default=7, type=int)
    parser.add_argument('--logdir', default='/home/i-lun/works/smt/tmp2', type=str)
    parser.add_argument('--batch_size', default=38, type=int)
    parser.add_argument('--decay_start', default=1000, type=int)
    parser.add_argument('--pattern', default='J0601_S', type=str)
    parser.add_argument('-tf', '--train_folder', default='/mnt/storage/P8_SMT/Connector/J0601/wuchi/split/train/OK/', type=str)
    parser.add_argument('--test_OK_folder', default='/mnt/storage/P8_SMT/Connector/J0601/wuchi/split/test/OK_and_NSG/', type=str)
    parser.add_argument('--test_NG_folder', default='/mnt/storage/P8_SMT/Connector/J0601/wuchi/split/test/NG/', type=str)
    parser.add_argument('--use_pins', dest='use_pins', action='store_true')
    parser.set_defaults(use_pins=False)
    parser.add_argument('--baseline', dest='baseline', action='store_true')
    parser.set_defaults(baseline=False)

    args = parser.parse_args()

    # args.baseline = True
    # checkpoint_name = 'no_use_pins/ed2_m6_l10.5/checkpoint-39.78889083862305-1821'
    checkpoint_name = 'checkpoint-700'
    checkpoint_path = os.path.join('/home/i-lun/works/smt/j0601/nsg_split', checkpoint_name)
    saved_in = os.path.join('/home/i-lun/works/smt/reports/j0601/nsg_split', checkpoint_name)

    region_tensors, energy_tensors, z = model_for_inference(args.pattern, args.encoded_dims, args.mixtures, args.use_pins, args.latent_dims, args.baseline)

    # save_reconstruction_images(checkpoint_path, region_tensors, saved_in, test_OK_folder=args.test_OK_folder, test_NG_folder=args.test_NG_folder)
    inference_fn = partial(inference, region_tensors=region_tensors, energy_tensors=energy_tensors, checkpoint=checkpoint_path, z=z)
    summary_report(inference_fn, saved_in=saved_in, train_folder=args.train_folder, test_OK_folder=args.test_OK_folder, test_NG_folder=args.test_NG_folder)


if __name__ == '__main__':
    main()
