import numpy as np
import tensorflow as tf

from config import cfg

class CapsConv(tf.keras.layers.Layer):
    ''' Capsule layer.
    Args:
        num_units: integer, the length of the output vector of a capsule.
        with_routing: boolean, this capsule is routing with the
                      lower-level layer capsule.
        num_outputs: the number of capsules in this layer.

    Returns:
        A 4-D tensor.
    '''
    def __init__(self, num_units, with_routing=True):
        super(CapsConv, self).__init__()
        self.num_units = num_units
        self.with_routing = with_routing

    def call(self, inputs, num_outputs, kernel_size=None, stride=None):
        self.num_outputs = num_outputs
        self.kernel_size = kernel_size
        self.stride = stride
        input_shape = tf.shape(inputs)

        if not self.with_routing:
            # Check the input shape using TensorFlow operations
            tf.debugging.assert_equal(input_shape, [cfg.batch_size, 20, 20, 256], 
                                    message="Input shape must be [cfg.batch_size, 20, 20, 256]")
            
            # the PrimaryCaps layer
            capsules = []
            for i in range(self.num_units):
                with tf.name_scope('ConvUnit_' + str(i)):
                    caps_i = tf.keras.layers.Conv2D(self.num_outputs,
                                                     self.kernel_size,
                                                     self.stride,
                                                     padding="valid")(inputs)
                    caps_i = tf.reshape(caps_i, shape=(cfg.batch_size, -1, 1, 1))
                    capsules.append(caps_i)

            assert capsules[0].get_shape() == [cfg.batch_size, 1152, 1, 1]

            # [batch_size, 1152, 8, 1]
            capsules = tf.concat(capsules, axis=2)
            capsules = squash(capsules)
            assert capsules.get_shape() == [cfg.batch_size, 1152, 8, 1]

        else:
            # the DigitCaps layer
            # Reshape the input into shape [batch_size, 1152, 8, 1]
            input_tensor = tf.reshape(inputs, shape=(cfg.batch_size, 1152, 8, 1))

            # b_IJ: [1, num_caps_l, num_caps_l_plus_1, 1]
            b_IJ = tf.zeros(shape=[1, 1152, 10, 1], dtype=tf.float32)
            capsules = []
            for j in range(self.num_outputs):
                with tf.name_scope('caps_' + str(j)):
                    caps_j, b_IJ = capsule(input_tensor, b_IJ, j)
                    capsules.append(caps_j)

            # Return a tensor with shape [batch_size, 5, 16, 1]
            capsules = tf.concat(capsules, axis=1)
            assert capsules.get_shape() == [cfg.batch_size, 5, 16, 1]

        return capsules

    
def capsule(input, b_IJ, idx_j):
    ''' The routing algorithm for one capsule in the layer l+1.
    
    Args:
        input: A Tensor with [batch_size, num_caps_l=1152, length(u_i)=8, 1]
               shape, num_caps_l meaning the number of capsule in the layer l.
    Returns:
        A Tensor of shape [batch_size, 1, length(v_j)=16, 1] representing the
        vector output `v_j` of capsule j in the layer l+1
    '''
    with tf.name_scope('routing'):
        w_initializer = np.random.normal(size=[1, 1152, 8, 16], scale=0.01)
        W_Ij = tf.Variable(w_initializer, dtype=tf.float32)
        W_Ij = tf.tile(W_Ij, [cfg.batch_size, 1, 1, 1])

        # Calculate u_hat
        u_hat = tf.matmul(W_Ij, input, transpose_a=True)

        shape = b_IJ.get_shape().as_list()
        size_splits = [idx_j, 1, shape[2] - idx_j - 1]
        for r_iter in range(cfg.iter_routing):
            c_IJ = tf.nn.softmax(b_IJ, axis=2)

            b_Il, b_Ij, b_Ir = tf.split(b_IJ, size_splits, axis=2)
            c_Il, c_Ij, b_Ir = tf.split(c_IJ, size_splits, axis=2)

            s_j = tf.multiply(c_Ij, u_hat)
            s_j = tf.reduce_sum(s_j, axis=1, keepdims=True)

            v_j = squash(s_j)

            # Tile v_j from [batch_size, 1, 16, 1] to [batch_size, 1152, 16, 1]
            v_j_tiled = tf.tile(v_j, [1, 1152, 1, 1])
            u_produce_v = tf.matmul(u_hat, v_j_tiled, transpose_a=True)

            b_Ij += tf.reduce_sum(u_produce_v, axis=0, keepdims=True)
            b_IJ = tf.concat([b_Il, b_Ij, b_Ir], axis=2)

        return v_j, b_IJ


def squash(vector):
    '''Squashing function.
    Args:
        vector: A 4-D tensor with shape [batch_size, num_caps, vec_len, 1],
    Returns:
        A 4-D tensor with the same shape as vector but
        squashed in 3rd and 4th dimensions.
    '''
    vec_abs = tf.sqrt(tf.reduce_sum(tf.square(vector), axis=-2, keepdims=True))  # a scalar
    scalar_factor = tf.square(vec_abs) / (1 + tf.square(vec_abs))
    vec_squashed = scalar_factor * tf.divide(vector, vec_abs + tf.keras.backend.epsilon())  # element-wise
    return vec_squashed
