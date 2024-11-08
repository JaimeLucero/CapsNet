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
flags.DEFINE_float('learning_rate', 0.001, 'learning rate for optimizer')  # Add this line


############################
#   environment setting    #
############################
flags.DEFINE_string('dataset', 'data/rice/', 'the path for dataset')
flags.DEFINE_boolean('is_training', True, 'train or predict phase')
flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing exampls')
flags.DEFINE_string('logdir', 'logdir', 'logs directory')

cfg = tf.compat.v1.app.flags.FLAGS
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
