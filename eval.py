import numpy as np
import tensorflow as tf

from config import cfg
from utils import load_custom_data
from utils import save_images
from capsNet import CapsNet

if __name__ == '__main__':
    # Load the CapsNet model
    capsNet = CapsNet(is_training=False)

    # Load the dataset
    test_dataset = load_custom_data(cfg.dataset, is_training=False)

    # Restore the latest checkpoint
    checkpoint_path = tf.train.latest_checkpoint(cfg.logdir)
    if checkpoint_path:
        capsNet.model.load_weights(checkpoint_path)
        tf.get_logger().info('Model weights restored from %s', checkpoint_path)
    else:
        tf.get_logger().warning('No checkpoint found at %s', cfg.logdir)

    reconstruction_err = []

    # Evaluate the model on the test dataset
    for step, (images, labels) in enumerate(test_dataset):
        recon_imgs = capsNet.decoded(images)  # Assuming decoded is a method of CapsNet
        orgin_imgs = tf.reshape(images, (cfg.batch_size, -1))  # Flatten the original images
        squared = tf.square(recon_imgs - orgin_imgs)
        reconstruction_err.append(tf.reduce_mean(squared).numpy())  # Convert tensor to numpy

        if step % 5 == 0:
            imgs = tf.reshape(recon_imgs, (cfg.batch_size, 28, 28, 1))
            size = 6
            save_images(imgs[0:size * size, :].numpy(), [size, size], f'results/test_{step:03d}.png')

    print('Test accuracy:')
    print((1. - np.mean(reconstruction_err)) * 100)
