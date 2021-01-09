"""
Paper

S. Rendle, “Factorization machines,” in Proceedings of IEEE International Conference on Data Mining (ICDM), pp. 995–1000, 2010

"""

import sys
import tensorflow as tf
from model.FM.preprocess import build_features
from utils import tf_estimator_model, add_layer_summary, build_estimator_helper

@tf_estimator_model
def model_fn(features, labels, mode, params):
    """
    FM model
    """
    feature_columns= build_features()

    input = tf.feature_column.input_layer(features, feature_columns)

    input_dim = input.get_shape().as_list()[-1]

    with tf.variable_scope('linear'):
        init = tf.random_normal( shape = (input_dim,1) )
        w = tf.get_variable('w', dtype = tf.float32, initializer = init, validate_shape = False)
        b = tf.get_variable('b', shape = [1], dtype= tf.float32)

        linear_term = tf.add(tf.matmul(input,w), b)
        add_layer_summary( linear_term.name, linear_term)

    with tf.variable_scope('fm_interaction'):
        init = tf.truncated_normal(shape = (input_dim, params['factor_dim']))
        v = tf.get_variable('v', dtype = tf.float32, initializer = init, validate_shape = False)

        sum_square = tf.pow(tf.matmul(input, v),2)
        square_sum = tf.matmul(tf.pow(input,2), tf.pow(v,2))

        interaction_term = 0.5 * tf.reduce_sum(sum_square - square_sum, axis=1, keep_dims= True)

        add_layer_summary(interaction_term.name, interaction_term)

    with tf.variable_scope('output'):
        y = tf.math.add(interaction_term, linear_term)
        add_layer_summary(y.name, y)

    return y



build_estimator = build_estimator_helper(
    model_fn = {
        'census':model_fn
    },
    params = {
        'census':{
            'learning_rate' :0.01,
            'factor_dim': 20
        }
    }
)
