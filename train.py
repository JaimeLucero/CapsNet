import tensorflow as tf
from tqdm import tqdm

from config import cfg
from capsNet import CapsNet

if __name__ == "__main__":
    # Create an instance of the CapsNet model
    capsNet = CapsNet(is_training=cfg.is_training)

    # Define the number of batches
    num_batch = int(52500 / cfg.batch_size)
    # Training loop
    for epoch in range(cfg.epoch):
        # Use tqdm to show progress
        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):

            # Perform a training step
            with tf.GradientTape() as tape:
                class_output, decoded = capsNet(capsNet.X)  # Forward pass
                loss_value = capsNet.train_loss(capsNet.Y, step)  # Calculate loss

            # Compute gradients and update weights
            gradients = tape.gradient(loss_value, capsNet.trainable_variables)
            capsNet.optimizer.apply_gradients(zip(gradients, capsNet.trainable_variables))

        # Optionally, save the model at the end of each epoch
        capsNet.save(f"{cfg.logdir}/model_epoch_{epoch:04d}_step_{capsNet.global_step.numpy():02d}.keras")

    tf.get_logger().info('Training done')
