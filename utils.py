import cv2
import numpy as np
import tensorflow as tf


def feed_image_to_tensors(image_file, region_tensors, current_batch):
    """
    :param image_file:
    :param region_tensors:
    :param current_batch:
    :return: the raw image with the channel orders of r, g, b
    """
    image = cv2.imread(image_file)
    b, g, r = cv2.split(image)
    rgb_img = cv2.merge([r, g, b])
    rgb_img = rgb_img.astype(np.float32) / 255.0

    for region_name in region_tensors:
        tensors = region_tensors[region_name]['tensors']
        sub_image = find_region(rgb_img, region_tensors[region_name], is_tf=False)
        current_batch[tensors].append(sub_image)

    return rgb_img


def find_region(image, region_def, is_tf=True):
    if region_def['region'] == 'all':
        return image
    elif type(region_def['region']) is tuple:
        xmin, ymin, xmax, ymax = region_def
        return image[ymin:ymax, xmin:xmax, :]
    elif type(region_def['region']) is list:
        assembles = None
        h = region_def['height']
        w = region_def['width']
        for rd in region_def['region']:
            xmin, ymin, xmax, ymax = rd
            sub_image = image[ymin:ymax, xmin:xmax, :]
            if assembles is None:
                assembles = sub_image
            else:
                if is_tf:
                    assembles = tf.concat([assembles, sub_image], axis=0)
                else:
                    assembles = np.concatenate([assembles, sub_image], axis=0)
        if is_tf:
            assembles = tf.image.resize_images(assembles, (h, w))
        else:
            assembles = cv2.resize(assembles, (w, h))
        return assembles
    else:
        return image

