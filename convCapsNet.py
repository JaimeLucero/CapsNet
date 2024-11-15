import tensorflow as tf
from config import cfg
from utils import get_batch_data
from capsLayer import CapsConv
from convCapsLayer import ConvCapsule

class ConvCapsNet(tf.keras.Model):
    def __init__(self, is_training=True):
        super(ConvCapsNet, self).__init__()
        self.summary_writer = tf.summary.create_file_writer('imgs/')  # Change the path as needed
        self.optimizer = tf.keras.optimizers.Adam()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        # Initialize empty layers; they will be built in `build()`
        self.use_reconstruction = cfg.use_reconstruction
        self.use_bias=cfg.use_bias
        self.make_skips = cfg.make_skips
        self.skip_dist = cfg.skip_dist

        # Set params
        self.dimensions = list(map(int, cfg.dimensions)) if cfg.dimensions != "" else []
        self.layersSize = list(map(int, cfg.layers)) if cfg.layers != "" else []

        if is_training:
            # Get the batches of data as a TensorFlow Dataset
            self.X, self.Y = get_batch_data()
            self.data = tf.data.Dataset.from_tensor_slices((self.X, self.Y)).batch(128)
            self.build_arch()
        else:
            self.build_arch()

        print('Setting up the main structure')

    def build_arch(self):
        # Conv1 Layer
        conv1_filters, conv1_kernel, conv1_stride = 256, 9, 1
        out_height = (28 - conv1_kernel) // conv1_stride + 1
        out_width = (28 - conv1_kernel) // conv1_stride + 1

        self.conv1 = tf.keras.layers.Conv2D(filters=conv1_filters, kernel_size=conv1_kernel,
                                            strides=conv1_stride, padding='valid', activation='relu')
        
        self.capsuleShape = tf.keras.layers.Reshape(target_shape=(out_height, out_width, 1, conv1_filters))

        self.capsule_layers = []
        for i, (capsules, dim) in enumerate(zip(self.layersSize[1:], self.dimensions[1:])):
            self.capsule_layer = ConvCapsule(
                name=f"ConvCapsuleLayer{i}",
                in_capsules=self.layersSize[i],
                in_dim=self.dimensions[i],
                out_capsules=capsules,
                out_dim=dim,
                kernel_size=3,
                routing_iterations=cfg.iterations,
                routing=cfg.routing  # Pass routing as a keyword argument
            )
            self.capsule_layers.append(self.capsule_layer)
            
        # Flatten and Fully Connected Capsules
        # Initialize the reshape layer
        self.flatten = tf.keras.layers.Reshape(target_shape=(20, 20, 256))

        # Define the PrimaryCaps layer
        self.primaryCaps = CapsConv(num_units=8, with_routing=False)

        # Define Final Capsule Layers
        self.fcCapsule = CapsConv(num_units=16, with_routing=True)

        if self.use_reconstruction:
            self.reconstruction_network = ReconstructionNetwork(
                name="ReconstructionNetwork",
                in_capsules=self.layersSize[-1], 
                in_dim=self.dimensions[-1],
                out_dim=28,
                img_dim=1)

        # Final classifier for class prediction
        self.classifier = tf.keras.layers.Dense(units=5, activation='softmax')

        self.residual = Residual()

    def call(self, x):
        # Initial convolutional layer
        x = self.conv1(x)
        x = self.capsuleShape(x)

        # Capsule layer forward pass with skip connections
        layers = []
        capsule_outputs = []    
        i = 0    
        for j, capsuleLayer in enumerate(self.capsule_layers):
            x = capsuleLayer(x)
            
            # Add skip connection
            capsule_outputs.append(x)
            if self.make_skips and i > 0 and i % self.skip_dist == 0:
                out_skip = capsule_outputs[j-self.skip_dist]
                if x.shape == out_skip.shape:
                    # Make residual connection
                    x = self.residual(x, out_skip)
                    i = -1
            
            i += 1
            layers.append(x)

        # Flatten the capsule outputs and pass through primary and final capsules
        x = self.flatten(x)
        x = self.primaryCaps(x, num_outputs=32, kernel_size=9, stride=2)
        x = self.fcCapsule(x, num_outputs=5)

        # Classification output
        self.v_length = tf.sqrt(tf.reduce_sum(tf.square(x), axis=2, keepdims=True) + 1e-2)
        self.softmax_v = tf.nn.softmax(self.v_length, axis=1)  # Softmax across the capsule dimension
        assert self.softmax_v.shape == [cfg.batch_size, 5, 1, 1], f"Expected shape [cfg.batch_size, 5, 1, 1], but got {self.softmax_v.shape}"
        
        class_output = tf.squeeze(self.softmax_v, axis=[2, 3])  # Squeeze dimensions 2 and 3 (size 1), but keep the class dimension
        assert class_output.shape == [cfg.batch_size, 5], f"Expected shape [cfg.batch_size, 5], but got {class_output.shape}"

        # Masking and reconstruction
        argmax_idx = tf.argmax(self.softmax_v, axis=1, output_type=tf.int32)  # Get the predicted class index
        assert argmax_idx.shape == [cfg.batch_size, 1, 1], f"Expected shape [cfg.batch_size, 1, 1], but got {argmax_idx.shape}"
        
        masked_v = []
        argmax_idx = tf.reshape(argmax_idx, shape=(cfg.batch_size, ))
        for batch_size in range(cfg.batch_size):
            v = x[batch_size][argmax_idx[batch_size], :]  # Get the capsule corresponding to the predicted class
            masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))  # Reshape and append to list

        self.masked_v = tf.concat(masked_v, axis=0)  # Concatenate masked vectors across the batch
        assert self.masked_v.shape == [cfg.batch_size, 1, 16, 1], f"Expected shape [cfg.batch_size, 1, 16, 1], but got {self.masked_v.shape}"

        # Decoder forward pass (using tf.keras.layers.Dense)
        vector_j = tf.reshape(self.masked_v, shape=(cfg.batch_size, -1))  # Flatten the masked vector
        fc1 = tf.keras.layers.Dense(units=512, activation='relu')(vector_j)  # Use Dense from tf.keras
        assert fc1.shape == [cfg.batch_size, 512], f"Expected shape [cfg.batch_size, 512], but got {fc1.shape}"

        fc2 = tf.keras.layers.Dense(units=1024, activation='relu')(fc1)  # Use Dense from tf.keras
        assert fc2.shape == [cfg.batch_size, 1024], f"Expected shape [cfg.batch_size, 1024], but got {fc2.shape}"

        self.decoded = tf.keras.layers.Dense(units=784, activation='sigmoid')(fc2)  # Use Dense from tf.keras
        assert self.decoded.shape == [cfg.batch_size, 784], f"Expected shape [cfg.batch_size, 784], but got {self.decoded.shape}"
        
        return class_output, self.decoded
        
    def train_loss(self, y_true, step):
        # 1. The margin loss
        # Ensure self.v_length has the expected shape: [batch_size, num_classes, 1, 1]
        assert self.v_length.shape == [cfg.batch_size, 5, 1, 1], f"Expected v_length shape: {[cfg.batch_size, 5, 1, 1]}, but got {self.v_length.shape}"
        
        max_l = tf.square(tf.maximum(0., cfg.m_plus - self.v_length))
        max_r = tf.square(tf.maximum(0., self.v_length - cfg.m_minus))
        
        # Ensure the shapes of max_l and max_r are correct
        assert max_l.shape == [cfg.batch_size, 5, 1, 1], f"Expected max_l shape: {[cfg.batch_size, 5, 1, 1]}, but got {max_l.shape}"
        assert max_r.shape == [cfg.batch_size, 5, 1, 1], f"Expected max_r shape: {[cfg.batch_size, 5, 1, 1]}, but got {max_r.shape}"

        # Reshape: [batch_size, 5, 1, 1] => [batch_size, 5]
        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))

        # 2. One-hot encode the true labels (ensure y_true is categorical)
        # Ensure y_true is one-hot encoded with shape [batch_size, num_classes]
        assert y_true.shape[1] == 5, f"Expected y_true shape to be [batch_size, 5], but got {y_true.shape}"

        T_c = tf.one_hot(tf.argmax(y_true, axis=1), depth=5)
        assert T_c.shape == [cfg.batch_size, 5], f"Expected T_c shape: {[cfg.batch_size, 5]}, but got {T_c.shape}"

        # Calculate the margin loss
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r
        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        # 3. The reconstruction loss
        # Ensure the shape of self.X is correct and reshaped to [batch_size, -1]
        assert self.X.shape == (cfg.batch_size, 28, 28, 1), f"Expected self.X shape: {(cfg.batch_size, 28, 28, 1)}, but got {self.X.shape}"
        
        original = tf.reshape(self.X, shape=(cfg.batch_size, -1))
        assert original.shape == [cfg.batch_size, 784], f"Expected original shape: {[cfg.batch_size, 784]}, but got {original.shape}"

        # Ensure self.decoded is of the correct shape
        assert self.decoded.shape == [cfg.batch_size, 784], f"Expected decoded shape: {[cfg.batch_size, 784]}, but got {self.decoded.shape}"

        squared = tf.square(self.decoded - original)
        self.reconstruction_err = tf.reduce_mean(squared)

        # 4. Total loss
        self.total_loss = self.margin_loss + 0.0005 * self.reconstruction_err

        # Assert that total_loss is a scalar
        assert self.total_loss.shape == (), f"Expected total_loss to be a scalar, but got shape {self.total_loss.shape}"
        
        tf.summary.trace_on(graph=True)
        # Summary
        with self.summary_writer.as_default():
            tf.summary.scalar('margin_loss', self.margin_loss, step=step)
            tf.summary.scalar('reconstruction_loss', self.reconstruction_err, step=step)
            tf.summary.scalar('total_loss', self.total_loss, step=step)
            recon_img = tf.reshape(self.decoded, shape=(cfg.batch_size, 28, 28, 1))
            tf.summary.image('reconstruction_img', recon_img, step=step)

        # Now, export the trace (this will capture the graph)
        tf.summary.trace_export(name="model_graph", step=step, profiler_outdir=cfg.logdir+'/'+cfg.model)

        return self.total_loss

    

layers = tf.keras.layers
models = tf.keras.models

class ReconstructionNetwork(tf.keras.Model):

    def __init__(self, in_capsules, in_dim, name="", out_dim=28, img_dim=1):
        super(ReconstructionNetwork, self).__init__(name=name)

        self.in_capsules = in_capsules
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.y = None

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(512, name="fc1", activation=tf.nn.relu)
        self.fc2 = layers.Dense(1024, name="fc2", activation=tf.nn.relu)
        self.fc3 = layers.Dense(out_dim * out_dim * img_dim, name="fc3", activation=tf.nn.sigmoid)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)  # Reconstruction output


class Residual(tf.keras.Model):
    def call(self, out_prev, out_skip):
        x = tf.keras.layers.Add()([out_prev, out_skip])
        #x = squash(x)
        return x

    def count_params(self):
        return 0
