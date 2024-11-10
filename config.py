import tensorflow as tf

flags = tf.compat.v1.app.flags


############################
#    hyper parameters      #
############################

# For separate margin loss
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('epoch', 1, 'epoch')
flags.DEFINE_integer('iter_routing', 3, 'number of iterations in routing algorithm')
flags.DEFINE_string('routing', 'rba', 'Routing algorithm (rba, em, sda)')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate for optimizer')  # Add this line
flags.DEFINE_float('iterations', 2, 'number of iteration for convCaps')  # Add this line
flags.DEFINE_list('dimensions', "8,8,16", 'Comma-separated list of layers. Each number represents the number of hidden units except for the first layer, the number of channels.')  # Add this line
flags.DEFINE_list('layers', "16,32,10", 'Comma-separated list of layers. Each number represents the dimension of the layer.')  # Add this line
flags.DEFINE_boolean('use_reconstruction', True, 'Use the reconstruction network as regularization loss')
flags.DEFINE_boolean('use_bias', True, 'Add a bias term to the preactivation')
flags.DEFINE_boolean('make_skips', True, 'Add a skip connection between of same shape.')
flags.DEFINE_integer('skip_dist', 2, 'Distance of skip connection')


############################
#   environment setting    #
############################
flags.DEFINE_string('dataset', 'data/rice/', 'the path for dataset')
flags.DEFINE_boolean('is_training', True, 'train or predict phase')
flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing exampls')
flags.DEFINE_string('logdir', 'logdir', 'logs directory')
flags.DEFINE_string('model', 'resCapsNet', 'set to capsNet or resCapsNet')

cfg = tf.compat.v1.app.flags.FLAGS
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
