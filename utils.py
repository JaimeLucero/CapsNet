import numpy as np
import tensorflow as tf
from PIL import Image
from config import cfg
from PIL import Image
import os
import matplotlib.pyplot as plt


import tensorflow as tf
import numpy as np
import os

# def load_custom_data(path, is_training):
#     # Read training images
#     with open(os.path.join(cfg.dataset, 'train-images-idx3-ubyte'), 'rb') as fd:
#         loaded = np.fromfile(file=fd, dtype=np.uint8)
#     trX = loaded[16:].reshape((52500, 28, 28, 1)).astype(float)  # Keep the 52500 size

#     # Read training labels
#     with open(os.path.join(cfg.dataset, 'train-labels-idx1-ubyte'), 'rb') as fd:
#         loaded = np.fromfile(file=fd, dtype=np.uint8)
#     trY = loaded[8:].reshape((52500)).astype(float)  # Keep the 52500 size

#     # Read test images
#     with open(os.path.join(cfg.dataset, 'test-images-idx3-ubyte'), 'rb') as fd:
#         loaded = np.fromfile(file=fd, dtype=np.uint8)
#     teX = loaded[16:].reshape((22500, 28, 28, 1)).astype(float)  # Keep the 22500 size

#     # Read test labels
#     with open(os.path.join(cfg.dataset, 'test-labels-idx1-ubyte'), 'rb') as fd:
#         loaded = np.fromfile(file=fd, dtype=np.uint8)
#     teY = loaded[8:].reshape((22500)).astype(float)  # Keep the 22500 size

#     # Normalize pixel values to [0, 1] and convert to tensor
#     trX = tf.convert_to_tensor(trX / 255., tf.float32)

#     # One-hot encode the labels for 5 classes (make sure this matches your dataset)
#     trY = tf.one_hot(trY, depth=5, axis=1, dtype=tf.float32)
#     teY = tf.one_hot(teY, depth=5, axis=1, dtype=tf.float32)

#     # Return training or testing data
#     if is_training:
#         return trX, trY
#     else:
#         return teX / 255.0, teY
    


def load_images_from_folder(folder_path):
    images = []
    labels = []
    class_names = sorted(os.listdir(folder_path))  # Sort to keep class names consistent

    # Filter out non-directory files (e.g., .DS_Store)
    class_names = [class_name for class_name in class_names if not class_name.startswith('.')]
    
    # Print the number of classes for verification
    print(f"Found {len(class_names)} classes: {class_names}")
    
    if len(class_names) != 5:
        raise ValueError(f"Expected 5 classes, but found {len(class_names)} classes in the dataset.")
    
    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(folder_path, class_name)
        
        if os.path.isdir(class_folder):
            for filename in os.listdir(class_folder):
                img_path = os.path.join(class_folder, filename)
                
                # Load the image
                img = Image.open(img_path).convert('L')  # Convert to grayscale if needed
                img = img.resize((28, 28))  # Resize to 28x28, similar to MNIST
                
                # Convert the image to numpy array and normalize
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=-1)  # Add the channel dimension (28, 28, 1)
                
                images.append(img_array)
                labels.append(label)  # Use the folder index as the label
    
    # Convert lists to numpy arrays
    return np.array(images), np.array(labels)


def load_custom_data(path, is_training):
    # Load train and test data
    train_images, train_labels = load_images_from_folder(f'{path}/train')
    test_images, test_labels = load_images_from_folder(f'{path}/test')

    # Normalize images to [0, 1] range and convert to tensors
    train_images = tf.convert_to_tensor(train_images, tf.float32)
    test_images = tf.convert_to_tensor(test_images, tf.float32)

    # One-hot encode labels for both training and test
    train_labels = tf.one_hot(train_labels, depth=5, axis=1, dtype=tf.float32)
    test_labels = tf.one_hot(test_labels, depth=5, axis=1, dtype=tf.float32)

    if is_training:
        return train_images, train_labels
    else:
        return test_images, test_labels
    
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


def visualize_reconstructed_images(decoded_images, num_images=10):
    """
    Display reconstructed images.
    
    Parameters:
    - decoded_images (np.array): Decoded output images, should be in (batch_size, height, width).
    - num_images (int): Number of images to display. Default is 10.
    """
    # Ensure decoded images are in numpy format
    decoded_images = np.array(decoded_images)
    decoded_images = np.clip(decoded_images, 0, 1)  # Clip to valid [0, 1] range if necessary

    # If images are flattened, reshape to (28, 28)
    if len(decoded_images.shape) == 2:  # Check if shape is (batch_size, 784)
        decoded_images = decoded_images.reshape((-1, 28, 28))  # Reshape to (batch_size, 28, 28)
    
    # Limit the number of images to display
    num_images = min(num_images, decoded_images.shape[0])
    
    # Plot the images in a grid format
    plt.figure(figsize=(10, 10))
    grid_size = int(np.ceil(np.sqrt(num_images)))  # Define grid based on number of images
    for i in range(num_images):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(decoded_images[i], cmap='gray')  # Use 'gray' for grayscale images
        plt.axis('off')  # Hide axes
    
    plt.suptitle("Reconstructed Images")
    plt.tight_layout()
    plt.show()

