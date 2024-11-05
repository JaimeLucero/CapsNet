import tensorflow as tf
from config import cfg
from utils import get_batch_data
from capsLayer import CapsConv

class CapsNet(tf.keras.Model):
    def __init__(self, is_training=True):
        super(CapsNet, self).__init__()
        self.summary_writer = tf.summary.create_file_writer('imgs/')  # Change the path as needed
        self.optimizer = tf.keras.optimizers.Adam()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        # Initialize empty layers; they will be built in `build()`
        self.conv1 = None
        self.primaryCaps = None
        self.digitCaps = None
        self.classifier = None

        if is_training:
            # Get the batches of data as a TensorFlow Dataset
            self.X, self.Y = get_batch_data()
            self.data = tf.data.Dataset.from_tensor_slices((self.X, self.Y)).batch(128)
            self.build_arch()
        else:
            self.X = tf.keras.Input(tf.float32, shape=(cfg.batch_size, 28, 28, 1))  # Use Input layer for shape definition
            self.build_arch()

        print('Setting up the main structure')

    def build_arch(self):
        # Define the Conv1 layer
        with tf.name_scope('Conv1_layer'):
            self.conv1 = tf.keras.layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid')

        # Define the PrimaryCaps layer
        with tf.name_scope('PrimaryCaps_layer'):
            self.primaryCaps = CapsConv(num_units=8, with_routing=False)

        # Define the DigitCaps layer
        with tf.name_scope('DigitCaps_layer'):
            self.digitCaps = CapsConv(num_units=16, with_routing=True)

        # Define the final dense layer for classification
        self.classifier = tf.keras.layers.Dense(units=5, activation='softmax')  # For 5 classes

    def call(self, inputs):
        # Forward pass through the layers
        x = self.conv1(inputs)  # Apply Conv1 layer
        x = self.primaryCaps(x, num_outputs=32, kernel_size=9, stride=2)  # Pass required arguments
        x = self.digitCaps(x, num_outputs=5)  # Pass required arguments

        # Classification output
        self.v_length = tf.sqrt(tf.reduce_sum(tf.square(x), axis=2, keepdims=True))
        self.softmax_v = tf.nn.softmax(self.v_length, axis=1)
        class_output = tf.squeeze(self.softmax_v)  # Shape: (batch_size, num_classes)

        # Masking and reconstruction
        # argmax_idx = tf.argmax(self.softmax_v, axis=1, output_type=tf.int32)
        # argmax_idx = tf.reshape(argmax_idx, shape=(tf.shape(inputs)[0],))

        # masked_v = []
        # for batch_size in range(tf.shape(inputs)[0]):
        #     v = x[batch_size][argmax_idx[batch_size], :]
        #     masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))

        # self.masked_v = tf.concat(masked_v, axis=0)

        # Masking and reconstruction
        argmax_idx = tf.argmax(self.softmax_v, axis=1, output_type=tf.int32)
        self.masked_v = tf.gather(x, argmax_idx, batch_dims=0)  # More efficient masking


        # Decoder forward pass
        vector_j = tf.reshape(self.masked_v, shape=(tf.shape(inputs)[0], -1))
        fc1 = tf.keras.layers.Dense(units=512, activation='relu')(vector_j)
        fc2 = tf.keras.layers.Dense(units=1024, activation='relu')(fc1)
        self.decoded = tf.keras.layers.Dense(units=784, activation='sigmoid')(fc2)

        return class_output, self.decoded  # Return both classification and reconstruction outputs
    
    def train_loss(self, y_true, step):
        # 1. The margin loss
        # [batch_size, 10, 1, 1]
        max_l = tf.square(tf.maximum(0., cfg.m_plus - self.v_length))
        max_r = tf.square(tf.maximum(0., self.v_length - cfg.m_minus))
        
        assert max_l.get_shape() == [cfg.batch_size, 5, 1, 1]

        # Reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))

        # Calculate T_c: [batch_size, 10]
        # One-hot encode the true labels (ensure y_true is categorical)
        T_c = tf.one_hot(tf.argmax(y_true, axis=1), depth=5)

        # Calculate the margin loss
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r
        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        # 2. The reconstruction loss
        # Reshape input to compare
        original = tf.reshape(self.X, shape=(cfg.batch_size, -1))  
        squared = tf.square(self.decoded - original)
        self.reconstruction_err = tf.reduce_mean(squared)

        # 3. Total loss
        self.total_loss = self.margin_loss + 0.0005 * self.reconstruction_err

        # Summary
        with self.summary_writer.as_default():
            tf.summary.scalar('margin_loss', self.margin_loss, step=step)  # Replace with actual step
            tf.summary.scalar('reconstruction_loss', self.reconstruction_err, step=step)
            tf.summary.scalar('total_loss', self.total_loss, step=step)
            recon_img = tf.reshape(self.decoded, shape=(cfg.batch_size, 28, 28, 1))
            tf.summary.image('reconstruction_img', recon_img, step=step)

        return self.total_loss
