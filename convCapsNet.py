import tensorflow as tf
import math

from capsLayer import CapsConv
from convCapsLayer import ConvCapsule
from capsLayer import CapsConv

layers = tf.keras.layers
models = tf.keras.models

class ConvCapsNet(tf.keras.Model):

    def __init__(self, args):
        super(ConvCapsNet, self).__init__()

        # Set params
        dimensions = list(map(int, args.dimensions.split(","))) if args.dimensions != "" else []
        layers = list(map(int, args.layers.split(","))) if args.layers != "" else []

        self.use_bias=args.use_bias
        self.use_reconstruction = args.use_reconstruction
        self.make_skips = args.make_skips
        self.skip_dist = args.skip_dist

        CapsuleType = {
            "rba": CapsConv,
        }

        if args.dataset == 'mnist':
            img_size = 24 
        elif args.dataset == 'cifar10':
            img_size = 32
        else:
            img_size = 28
            # raise NotImplementedError()

        conv1_filters, conv1_kernel, conv1_stride = 128, 7, 2
        out_height = (img_size - conv1_kernel) // conv1_stride + 1
        out_width = (img_size - conv1_kernel) // conv1_stride + 1 

        with tf.name_scope(self.name):

            # normal convolution
            self.conv_1 = tf.keras.layers.Conv2D(
                        conv1_filters, 
                        kernel_size=conv1_kernel, 
                        strides=conv1_stride, 
                        padding='valid', 
                        activation="relu", 
                        name="conv1")

            # reshape into capsule shape
            self.capsuleShape = tf.keras.layers.Reshape(target_shape=(out_height, out_width, 1, conv1_filters), name='toCapsuleShape')

            self.capsule_layers = []
            for i in range(len(layers)-1):
                self.capsule_layers.append(
                    ConvCapsule(
                            name="ConvCapsuleLayer" + str(i), 
                            in_capsules=layers[i], 
                            in_dim=dimensions[i], 
                            out_dim=dimensions[i], 
                            out_capsules=layers[i+1], 
                            kernel_size=3,
                            routing_iterations=args.iterations,
                            routing=args.routing))

            # flatten for input to FC capsule
            self.flatten = tf.keras.layers.Reshape(target_shape=(out_height * out_width * layers[-2], dimensions[-2]), name='flatten')
            
            # fully connected caspule layer
            self.fcCapsuleLayer = CapsConv(
                        name="FCCapsuleLayer",
                        in_capsules = out_height * out_width * layers[-2], 
                        in_dim = dimensions[-2], 
                        out_capsules = layers[-1],
                        out_dim = dimensions[-1], 
                        use_bias = self.use_bias)                      


            if self.use_reconstruction:
                self.reconstruction_network = ReconstructionNetwork(
                    name="ReconstructionNetwork",
                    in_capsules=layers[-1], 
                    in_dim=dimensions[-1],
                    out_dim=args.img_height,
                    img_dim=args.img_depth)

            self.norm = Norm()
            self.residual = Residual()


    # Inference
    def call(self, x, y):
        x = self.conv_1(x)
        x = self.capsuleShape(x)

        print('x: ', x.shape)

        layers = []
        capsule_outputs = []    
        i = 0    
        for j, capsuleLayer in enumerate(self.capsule_layers):
            x = capsuleLayer(x)
            
            # add skip connection
            capsule_outputs.append(x)
            if self.make_skips and i > 0 and i % self.skip_dist == 0:
                out_skip = capsule_outputs[j-self.skip_dist]
                if x.shape == out_skip.shape:
                    #print('make residual connection from ', j-self.skip_dist, ' to ', j)
                    x = self.residual(x, out_skip)
                    i = -1
            
            i += 1
            layers.append(x)

        x = self.flatten(x)
        x = self.fcCapsuleLayer(x)
 
        r = self.reconstruction_network(x, y) if self.use_reconstruction else None
        out = self.norm(x)

        return out, r, layers
    
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
        self.fc3 = layers.Dense(out_dim * out_dim * img_dim, name="fc3", activation=tf.sigmoid)


    def call(self, x, y):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
class Residual(tf.keras.Model):
    def call(self, out_prev, out_skip):
        x = tf.keras.layers.Add()([out_prev, out_skip])
        #x = squash(x)
        return x

    def count_params(self):
        return 0

class Norm(tf.keras.Model):
    def call(self, inputs):
        x = tf.norm(inputs, name="norm", axis=-1)
        return x