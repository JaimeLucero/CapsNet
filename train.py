import tensorflow as tf
from tqdm import tqdm

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
    # Training loop
    for epoch in range(cfg.epoch):
        # Use tqdm to show progress
        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):

            # Perform a training step
            with tf.GradientTape() as tape:
                if cfg.model == 'capsNet':
                    class_output, decoded = model(model.X)  # Forward pass
                    loss_value = model.train_loss(model.Y, step)  # Calculate loss
                else:
                    class_output, decoded, layer = model(model.X)  # Forward pass
                    loss_value = model.train_loss(model.Y, step)  # Calculate loss

            # Compute gradients and update weights
            gradients = tape.gradient(loss_value, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Optionally, save the model at the end of each epoch
        model.save_weights(f"{cfg.logdir}/{cfg.model}/model_weights_epoch_{epoch:04d}_step_{model.global_step.numpy():02d}.weights.h5")

    tf.get_logger().info('Training done')
