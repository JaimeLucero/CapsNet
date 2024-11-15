import tensorflow as tf
from config import cfg
from utils import get_batch_data, visualize_reconstructed_images
from capsLayer import CapsConv

class CapsNet(tf.keras.Model):
    def __init__(self, is_training=True):
        super(CapsNet, self).__init__()
        self.summary_writer = tf.summary.create_file_writer(F'summary/{cfg.model}/') 
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Define layers in __init__ instead of build for simpler organization
        self.conv1 = tf.keras.layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid',
                                            kernel_initializer='he_normal')
        self.primaryCaps = CapsConv(num_units=8, with_routing=False)
        self.digitCaps = CapsConv(num_units=16, with_routing=True)
        self.classifier = tf.keras.layers.Dense(units=5, activation='softmax')

        # Decoder layers
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(784, activation='sigmoid')
        ])

        # Data setup
        if is_training:
            self.X, self.Y = get_batch_data()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.primaryCaps(x, num_outputs=32, kernel_size=9, stride=2)
        x = self.digitCaps(x, num_outputs=5)

        # Vector length and softmax classification
        self.v_length = tf.sqrt(tf.reduce_sum(tf.square(x), axis=2, keepdims=True))
        class_output = tf.nn.softmax(self.v_length, axis=1)

        # Reconstruction through masked vectors
        argmax_idx = tf.argmax(class_output, axis=1)
        masked_v = tf.gather(x, argmax_idx, batch_dims=1)
        vector_j = tf.reshape(masked_v, shape=(cfg.batch_size, -1))
        # Normalize the vector_j (scale the length to 1)
        vector_j = vector_j / tf.norm(vector_j, axis=1, keepdims=True)

        self.decoded = self.decoder(vector_j)

        # visualize_reconstructed_images(self.decoded)

        return class_output, self.decoded

    def train_loss(self, y_true, step):
        # One-hot encoding of true labels for margin loss
        T_c = tf.one_hot(tf.argmax(y_true, axis=1), depth=5)

        # Margin and reconstruction loss
        max_l = tf.square(tf.maximum(0., cfg.m_plus - self.v_length))
        max_r = tf.square(tf.maximum(0., self.v_length - cfg.m_minus))
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r
        margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        # Reconstruction loss
        original = tf.reshape(self.X, shape=(cfg.batch_size, -1))
        reconstruction_err = tf.reduce_mean(tf.square(self.decoded - original))
        
        # Total loss
        total_loss = margin_loss + 0.0005 * reconstruction_err

        # Logging
        with self.summary_writer.as_default():
            tf.summary.scalar('total_loss', total_loss, step=step)
            tf.summary.image('reconstruction_img', tf.reshape(self.decoded, shape=(cfg.batch_size, 28, 28, 1)), step=step)

        return total_loss, margin_loss, reconstruction_err

