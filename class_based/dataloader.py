"""Data loading scripts"""
## NOTE: You do not need to modify this block but you will need to use it.
import numpy as np
import os
import random
import tensorflow as tf
from scipy import misc
import imageio


def get_images(paths, labels, n_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
      paths: A list of character folders
      labels: List or numpy array of same length as paths
      n_samples: Number of images to retrieve per character
    Returns:
      List of (label, image_path) tuples
    """
    if n_samples is not None:
        sampler = lambda x: random.sample(x, n_samples)
    else:
        sampler = lambda x: x
    images_labels = [(i, os.path.join(path, image))
                     for i, path in zip(labels, paths)
                     for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


def image_file_to_array(filename, dim_input):
    """
    Takes an image path and returns numpy array
    Args:
      filename: Image filename
      dim_input: Flattened shape of image
    Returns:
      1 channel image
    """
    image = imageio.imread(filename)
    image = image.reshape([dim_input])
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image
    return image


class DataGenerator(object):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(self, num_classes, num_samples_per_class, num_meta_test_classes, num_meta_test_samples_per_class,
                 num_train_chars=1100, num_val_chars=100,
                 config={}):
        """
        Args:
          num_classes: Number of classes for classification (K-way)
          num_samples_per_class: num samples to generate per class in one batch
          num_meta_test_classes: Number of classes for classification (K-way) at meta-test time
          num_meta_test_samples_per_class: num samples to generate per class in one batch at meta-test time
          batch_size: size of meta batch size (e.g. number of functions)
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes
        self.num_meta_test_samples_per_class = num_meta_test_samples_per_class
        self.num_meta_test_classes = num_meta_test_classes

        data_folder = config.get('data_folder', './omniglot_resized')
        self.img_size = config.get('img_size', (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))]

        # random.seed(123)
        # random.shuffle(character_folders)
        num_val = num_val_chars
        num_train = num_train_chars
        self.metatrain_character_folders = character_folders[: num_train]
        self.metaval_character_folders = character_folders[
                                         num_train:num_train + num_val]
        self.metatest_character_folders = character_folders[
                                          num_train + num_val:]

    def sample_batch(self, batch_type, batch_size, shuffle=False, swap=False, fixed_folder_order=True):
        """
        Samples a batch for training, validation, or testing
        Args:
          batch_type: meta_train/meta_val/meta_test
          shuffle: randomly shuffle classes or not
          swap: swap number of classes (N) and number of samples per class (K) or not
          fixed_folder_order: Keeps all characters in fixed order. Tasks no longer mutually exclusive.
        Returns:
          A a tuple of (1) Image batch and (2) Label batch where
          image batch has shape [B, N, K, 784] and label batch has shape [B, N, K, N] if swap is False
          where B is batch size, K is number of samples per class, N is number of classes
        """
        if batch_type == "meta_train":
            folders = self.metatrain_character_folders
            num_classes = self.num_classes
            num_samples_per_class = self.num_samples_per_class
        elif batch_type == "meta_val":
            folders = self.metaval_character_folders
            num_classes = self.num_classes
            num_samples_per_class = self.num_samples_per_class
        else:
            folders = self.metatest_character_folders
            num_classes = self.num_meta_test_classes
            num_samples_per_class = self.num_meta_test_samples_per_class
        all_image_batches, all_label_batches = [], []
        for i in range(batch_size):
            ids = np.random.choice(list(range(len(folders))), num_classes, replace=False)
            if fixed_folder_order:
                ids = sorted(ids)
            sampled_character_folders = [folders[folder_id] for folder_id in ids]
            labels_and_images = get_images(sampled_character_folders, range(
                num_classes), n_samples=num_samples_per_class, shuffle=False)
            labels = [li[0] for li in labels_and_images]
            images = [image_file_to_array(
                li[1], self.dim_input) for li in labels_and_images]
            images = np.stack(images)
            labels = np.array(labels).astype(np.int32)
            labels = np.reshape(
                labels, (num_classes, num_samples_per_class))
            labels = np.eye(num_classes, dtype=np.float32)[labels]
            images = np.reshape(
                images, (num_classes, num_samples_per_class, -1))

            batch = np.concatenate([labels, images], 2)
            if shuffle:
                for p in range(num_samples_per_class):
                    np.random.shuffle(batch[:, p])

            labels = batch[:, :, :num_classes]
            images = batch[:, :, num_classes:]

            if swap:
                labels = np.swapaxes(labels, 0, 1)
                images = np.swapaxes(images, 0, 1)

            all_image_batches.append(images)
            all_label_batches.append(labels)
        all_image_batches = np.stack(all_image_batches)
        all_label_batches = np.stack(all_label_batches)
        return all_image_batches, all_label_batches


def get_datagen_as_dataset(data_generator: DataGenerator, batch_type, batch_size):

    def gen_fn():
        while True:
            yield data_generator.sample_batch(batch_type, batch_size)

    B = batch_size
    N = data_generator.num_classes
    K = data_generator.num_samples_per_class
    generator_types = (tf.float32, tf.int8)
    generator_shapes = tf.TensorShape([B, N, K, data_generator.dim_input]), tf.TensorShape([B, N, K, N])

    dataset = tf.data.Dataset.from_generator(gen_fn, generator_types)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
