import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from capsNet import CapsNet
from convCapsNet import ConvCapsNet
from config import cfg
from PIL import Image
import os
import glob

# Class labels mapping (example)
class_labels = {0: "Arborio", 1: "Basmati", 3: "Ipsala", 4: "Jasmine", 5: "Karacadag"}  # Update with your actual labels

# Load the trained model with the latest weights
def load_model():
    if cfg.model == 'capsNet':
        model = CapsNet(is_training=False)
    else:
        model = ConvCapsNet(is_training=False)
    
    weight_files = glob.glob(f"{cfg.logdir}/{cfg.model}/*.weights.h5")
    if not weight_files:
        raise FileNotFoundError("No weights files found.")
    
    latest_weight_file = max(weight_files, key=os.path.getctime)
    print(f"Loading weights from {latest_weight_file}")
    
    model.load_weights(latest_weight_file)
    return model

# Preprocess the input image to match model's input size
def preprocess_image(img_path):
    img = Image.open(img_path).convert('L')
    img = img.resize((28, 28))  # Adjust if necessary
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=-1)  # Add channel dimension for grayscale
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Make prediction and show reconstructed image
def predict(model, img_path):
    img = preprocess_image(img_path)
    class_output, decoded = model(img, training=False)
    
    # Ensure predicted_class is an integer
    predicted_class = np.argmax(class_output, axis=1)[0].item()
    
    # Decode the predicted class index to the label name
    predicted_label = class_labels.get(predicted_class, "Unknown")
    
    # Display the original image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img[0, :, :, 0], cmap='gray')
    plt.title(f"Original Image\nPredicted Class: {predicted_label}")
    plt.axis('off')
    
    # Display the reconstructed image from the model
    plt.subplot(1, 2, 2)
    reconstructed_img = decoded[0].numpy().reshape(28, 28)  # Reshape if needed to match output format
    plt.imshow(reconstructed_img, cmap='gray')
    plt.title("Reconstructed Image")
    plt.axis('off')
    
    plt.show()
    
    return predicted_label

# Main function to upload an image and predict
if __name__ == "__main__":
    model = load_model()

    # Specify image path
    img_path = "data_test/rice test 3.jpg"  # Replace with your image path

    # Predict on new image
    predicted_label = predict(model, img_path)
    print(f"The model predicted class: {predicted_label}")
