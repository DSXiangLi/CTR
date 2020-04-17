"""
Paper

S. Rendle, “Factorization machines,” in Proceedings of IEEE International Conference on Data Mining (ICDM), pp. 995–1000, 2010

"""

import sys
import tensorflow as tf
from preprocess import build_features
from utils import tf_estimator_model, add_layer_summary

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

        interaction_term = 0.5 * tf.reduce_mean(sum_square - square_sum, axis=1, keep_dims= True)

        add_layer_summary(interaction_term.name, interaction_term)

    with tf.variable_scope('output'):
        y = tf.math.add(interaction_term, linear_term)
        add_layer_summary(y.name, y)

    return y

def build_estimator(model_dir):

    run_config = tf.estimator.RunConfig(
        save_summary_steps=50,
        log_step_count_steps=50,
        keep_checkpoint_max = 3,
        save_checkpoints_steps =50
    )

    estimator = tf.estimator.Estimator(
        model_fn = model_fn,
        config = run_config,
        params = {
            'learning_rate' :0.01,
            'factor_dim': 20,
        },
        model_dir= model_dir
    )

    return estimator

