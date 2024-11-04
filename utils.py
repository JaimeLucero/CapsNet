import numpy as np
import tensorflow as tf
from PIL import Image
from config import cfg
import gzip
import pickle
import scipy


if __name__ == '__main__':
    X, Y = load_custom_data(cfg.dataset, cfg.is_training)
    print(X.get_shape())
    print(X.dtype)



def load_custom_data(dataset_path, is_training=True, batch_size=cfg.batch_size):
    # Load the compressed data
    with gzip.open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    # Assuming the data contains 'train_images', 'train_labels', 'test_images', 'test_labels'
    train_images = data['train_images']
    train_labels = data['train_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

    # Normalize the images to the range [0, 1]
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # Reshape the images to [num_samples, height, width, channels]
    # Assuming train_images has a shape of [52500, 28, 28] and needs to be reshaped to [52500, 28, 28, 1]
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

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

    # Check the shape of a single batch and unpack the data
    for images, labels in dataset.take(1):
        print("Loaded data shapes:", images.shape, labels.shape)  # Should be (batch_size, 28, 28, 1) and (batch_size, 10)
    
    # Return images and labels separately
    return images, labels

def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs
