import os
import numpy as np
import tensorflow as tf
from PIL import Image
from config import cfg
import gzip
import pickle

if __name__ == '__main__':
    X, Y = load_custom_data(cfg.dataset, cfg.is_training)


def load_custom_data(dataset_path, is_training=True, batch_size=cfg.batch_size):

    # Path to your compressed dataset
    compressed_file_path = dataset_path

    # Load the compressed data
    with gzip.open(compressed_file_path, 'rb') as f:
        data = pickle.load(f)

    # Assuming the data contains 'train_images', 'train_labels', 'test_images', 'test_labels'
    train_images = data['train_images']
    train_labels = data['train_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    # Shuffle, batch, and prefetch the training dataset for optimal performance
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    if is_training:
        return train_dataset
    else:
        return test_dataset

def get_batch_data():
    dataset = load_custom_data(cfg.dataset, is_training=True, batch_size=cfg.batch_size)

    # Check the shape of a single batch
    for images, labels in dataset.take(1):
        print("Loaded data shapes:", images.shape, labels.shape)  # Should be (batch_size, 28, 28, 1) and (batch_size, 10)
    
    return dataset
