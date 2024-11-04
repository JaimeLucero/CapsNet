import tensorflow as tf
from tqdm import tqdm

from config import cfg
from capsNet import CapsNet


if __name__ == "__main__":
    capsNet = CapsNet(is_training=cfg.is_training)
    tf.logging.info('Graph loaded')
    sv = tf.train.Supervisor(graph=capsNet.graph,
                             logdir=cfg.logdir,
                             save_model_secs=0)
    
    # Define the optimizer
    optimizer = tf.keras.optimizers.Adam()

    # Set up a checkpoint callback
    checkpoint_path = cfg.logdir + "/model_epoch_{epoch:04d}.h5"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_freq='epoch',  # Save at the end of each epoch
        verbose=1
    )

    # Training loop
    for epoch in range(cfg.epoch):
        total_loss = 0
        num_batches = 0

        # Iterate over the dataset
        for images, labels in tqdm(capsNet.X, ncols=70, leave=False, unit='b'):
            with tf.GradientTape() as tape:
                # Forward pass
                predictions = capsNet(images, training=True)
                # Calculate loss
                loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
                total_loss += loss
                num_batches += 1

            # Calculate gradients and update weights
            grads = tape.gradient(loss, capsNet.trainable_variables)
            optimizer.apply_gradients(zip(grads, capsNet.trainable_variables))

        # Average loss for the epoch
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch + 1} - Loss: {avg_loss.numpy()}')

        # Save model weights using the checkpoint callback
        checkpoint_callback.on_epoch_end(epoch)


    tf.logging.info('Training done')
