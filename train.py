import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import cfg
from capsNet import CapsNet
from convCapsNet import ConvCapsNet

# Clear any previous session
tf.keras.backend.clear_session()

if __name__ == "__main__":
    if cfg.model == 'capsNet':
        # Create an instance of the CapsNet model
        model = CapsNet(is_training=cfg.is_training)
    else:
        model = ConvCapsNet(is_training=cfg.is_training)

    # Define the number of batches
    num_batch = int(52500 / cfg.batch_size)

    # Initialize lists to track losses
    epoch_batch_losses = []
    epoch_margin_losses = []
    epoch_reconstruction_losses = []

    # Training loop
    for epoch in range(cfg.epoch):
        print('Epoch: ', epoch)
        epoch_loss = 0.0  # Track the total loss for the current epoch
        epoch_margin_loss = 0.0  # Track the margin loss for the current epoch
        epoch_reconstruction_loss = 0.0  # Track the reconstruction loss for the current epoch

        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            # Perform a training step
            with tf.GradientTape() as tape:
                if cfg.model == 'capsNet':
                    model(model.X)  # Forward pass
                    loss_value, margin_loss, reconstruction_err = model.train_loss(model.Y, step)  # Calculate total, margin, and reconstruction loss
                else:
                    model(model.X)  # Forward pass
                    loss_value, margin_loss, reconstruction_err = model.train_loss(step)  # Calculate total, margin, and reconstruction loss

            # Check if loss is NaN
            if tf.reduce_any(tf.math.is_nan(loss_value)):
                print(f"NaN detected at step {step}. Breaking the loop.")
                break  # Exit the loop if NaN is detected

            # Compute gradients
            grads = tape.gradient(loss_value, model.trainable_variables)
            for grad, var in zip(grads, model.trainable_variables):
                if tf.reduce_any(tf.math.is_nan(grad)):
                    print(f"NaN in gradient for layer {var.name}")
                    break

            # clipped_grads = [tf.clip_by_norm(g, clip_norm=1.0) if g is not None else None for g in grads]
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Increment global step
            model.global_step.assign_add(1)

            # Accumulate losses for the current batch
            epoch_loss += loss_value.numpy()
            epoch_margin_loss += margin_loss.numpy()
            epoch_reconstruction_loss += reconstruction_err.numpy()

        # Calculate average losses for the epoch
        avg_epoch_loss = epoch_loss / num_batch
        avg_margin_loss = epoch_margin_loss / num_batch
        avg_reconstruction_loss = epoch_reconstruction_loss / num_batch

        # Print the average loss for the epoch
        print(f'Epoch {epoch} - Avg Total Loss: {avg_epoch_loss}, Avg Margin Loss: {avg_margin_loss}, Avg Reconstruction Loss: {avg_reconstruction_loss}')

        # Append the average losses to the lists for plotting
        epoch_batch_losses.append(avg_epoch_loss)
        epoch_margin_losses.append(avg_margin_loss)
        epoch_reconstruction_losses.append(avg_reconstruction_loss)

        # Optionally, save the model at the end of each epoch
        model.save_weights(f"{cfg.logdir}/{cfg.model}/model_weights_epoch_{epoch:04d}_step_{model.global_step.numpy():02d}.weights.h5")

    # Plotting the training losses by epoch - Total Loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(cfg.epoch), epoch_batch_losses, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{cfg.model} Epoch-wise Total Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"imgs/{cfg.model}/epoch_total_loss.png")
    plt.close()

    # Plotting the margin loss by epoch
    plt.figure(figsize=(10, 6))
    plt.plot(range(cfg.epoch), epoch_margin_losses, label='Margin Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{cfg.model} Epoch-wise Margin Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"imgs/{cfg.model}/epoch_margin_loss.png")
    plt.close()

    # Plotting the reconstruction loss by epoch
    plt.figure(figsize=(10, 6))
    plt.plot(range(cfg.epoch), epoch_reconstruction_losses, label='Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{cfg.model} Epoch-wise Reconstruction Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"imgs/{cfg.model}/epoch_reconstruction_loss.png")
    plt.close()

    tf.get_logger().info('Training done')
