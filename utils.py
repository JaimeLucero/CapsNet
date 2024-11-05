import numpy as np
import tensorflow as tf
from PIL import Image
from config import cfg
from PIL import Image
import os



if __name__ == '__main__':
    X, Y = load_custom_data(cfg.dataset, cfg.is_training)
    print(X.get_shape())
    print(X.dtype)

def load_custom_data(path, is_training):
    fd = open(os.path.join(cfg.dataset, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((52500, 28, 28, 1)).astype(float)

    fd = open(os.path.join(cfg.dataset, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((52500)).astype(float)

    fd = open(os.path.join(cfg.dataset, 'test-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((22500, 28, 28, 1)).astype(float)

    fd = open(os.path.join(cfg.dataset, 'test-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((22500)).astype(float)

    # normalization and convert to a tensor [60000, 28, 28, 1]
    trX = tf.convert_to_tensor(trX / 255., tf.float32)

    # => [num_samples, 10]
    trY = tf.one_hot(trY, depth=10, axis=1, dtype=tf.float32)
    teY = tf.one_hot(teY, depth=10, axis=1, dtype=tf.float32)

    if is_training:
        return trX, trY
    else:
        return teX / 255., teY


    
def get_batch_data():
    trX, trY = load_custom_data(cfg.dataset, cfg.is_training)
    # Create a TensorFlow Dataset from the training data
    dataset = tf.data.Dataset.from_tensor_slices((trX, trY))
    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=len(trX), reshuffle_each_iteration=True)
    # Batch the dataset
    dataset = dataset.batch(cfg.batch_size)
    # Prefetch to improve performance
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    # Create an iterator
    iterator = dataset.as_numpy_iterator()
    # Get the first batch
    batch_X, batch_Y = next(iterator)
    return batch_X, batch_Y


def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width, channels]
        size: a list with two int elements, [rows, cols]
        path: the path to save the images
    '''
    # Normalize the images to the range [0, 255]
    imgs = (imgs + 1.0) * 127.5  # Convert to [0, 255] range
    imgs = np.clip(imgs, 0, 255).astype(np.uint8)
    
    # Merge images into a single image
    merged_img = mergeImgs(imgs, size)
    
    # Ensure the merged image has 3 channels (RGB)
    if merged_img.shape[2] == 1:  # Grayscale image
        merged_img = np.squeeze(merged_img, axis=2)  # Remove single channel dimension
        img_pil = Image.fromarray(merged_img, mode='L')  # 'L' mode for grayscale
    else:  # RGB image
        img_pil = Image.fromarray(merged_img, mode='RGB')
    
    img_pil.save(path)

def mergeImgs(images, size):
    h, w, c = images.shape[1], images.shape[2], images.shape[3]
    imgs = np.zeros((h * size[0], w * size[1], c), dtype=np.uint8)
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs
