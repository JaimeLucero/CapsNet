import numpy as np
import tensorflow as tf
from tqdm import tqdm  # Import tqdm
import matplotlib.pyplot as plt  # Import matplotlib for plotting

from config import cfg
from utils import load_custom_data, visualize_reconstructed_images
from utils import save_images
from capsNet import CapsNet
from convCapsNet import ConvCapsNet
import os 

if __name__ == '__main__':
    # Initialize the CapsNet model
    if cfg.model == 'capsNet':
        # Create an instance of the CapsNet model
        model = CapsNet(is_training=False)
    else:
        model = ConvCapsNet(is_training=False)

    # Load the test data
    teX, teY = load_custom_data(cfg.dataset, is_training=False)

    # Check for weights files in the log directory
    weights_files = [f for f in os.listdir(os.path.join(cfg.logdir, cfg.model)) if f.endswith('.weights.h5')]
    if not weights_files:
        print("No weights files found in the specified log directory.")
        exit()

    # Sort the files and get the latest one
    latest_weights_file = sorted(weights_files)[-1]  # Get the most recent weights file
    latest_weights_path = os.path.join(cfg.logdir, cfg.model, latest_weights_file)

    # Load the model weights
    model.load_weights(latest_weights_path)
    print(f"Model weights loaded from {latest_weights_path}")

    reconstruction_err = []

    # Evaluate the model on the test dataset with tqdm for progress tracking
    num_batches = 22500 // cfg.batch_size
    for i in tqdm(range(num_batches), total=num_batches, ncols=70, leave=False, unit='batch'):
        start = i * cfg.batch_size
        end = start + cfg.batch_size
        
        # Get the model outputs
        if cfg.model == 'capsNet':
            class_output, recon_imgs = model(teX[start:end])  # Unpack outputs
        else:
            class_output, recon_imgs, layers = model(teX[start:end])

        # Reshape original images if needed
        orgin_imgs = np.reshape(teX[start:end], (cfg.batch_size, 28, 28, 1))  # Ensure original images are in the correct shape

        # Ensure recon_imgs and orgin_imgs have the same shape for comparison
        if recon_imgs.shape != orgin_imgs.shape:
            recon_imgs = np.reshape(recon_imgs, orgin_imgs.shape)  # Adjust shape if necessary

        # Calculate the reconstruction error
        squared = np.square(recon_imgs - orgin_imgs)
        batch_reconstruction_err = np.mean(squared)
        reconstruction_err.append(batch_reconstruction_err)  # Accumulate error for each batch
        # Save images every 5 iterations
        if i % 5 == 0:
            recon_imgs_rescaled = np.uint8(recon_imgs * 255)  # Scale to [0, 255] for saving
            imgs = np.reshape(recon_imgs_rescaled, (cfg.batch_size, 28, 28, 1))
            size = 5
            save_images(imgs[0:size * size, :], [size, size], f'results/{cfg.model}/test_{i:03d}.png')

    # Calculate and print test accuracy
    mean_reconstruction_err = np.mean(reconstruction_err)
    test_accuracy = (1. - mean_reconstruction_err) * 100
    print('Test accuracy:')
    print(test_accuracy)

    # Plotting the reconstruction error over the batches
    plt.figure(figsize=(10, 6))
    plt.plot(reconstruction_err, label='Reconstruction Error')
    plt.xlabel('Batch Number')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error per Batch')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(f"{cfg.logdir}/{cfg.model}/reconstruction_error_per_batch.png")
    plt.close()
