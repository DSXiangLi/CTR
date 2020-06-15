"""
paper

Ying Shan, T. Ryan Hoens, 2016, Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features

"""


import tensorflow as tf
import numpy as np
from const import *
from model.DeepCrossing.preprocess import build_features
from utils import tf_estimator_model, add_layer_summary, build_estimator_helper

def residual_layer(x0, unit, dropout_rate, batch_norm, mode):
    # f(x): input_size -> unit -> input_size
    # output = relu(f(x) + x)
    input_size = x0.get_shape().as_list()[-1]

    # input_size -> unit
    x1 = tf.layers.dense(x0, units = unit, activation = 'relu')
    if batch_norm:
        x1 = tf.layers.batch_normalization( x1, center=True, scale=True,
                                               trainable=True,
                                               training=(mode == tf.estimator.ModeKeys.TRAIN) )
    if dropout_rate > 0:
        x1 = tf.layers.dropout( x1, rate=dropout_rate,
                               training=(mode == tf.estimator.ModeKeys.TRAIN) )
    # unit -> input_size
    x2 = tf.layers.dense(x1, units = input_size )
    # stack with original input and apply relu
    output = tf.nn.relu(tf.add(x2, x0))

    return output


@tf_estimator_model
def model_fn(features, labels, mode, params):
    dense_feature = build_features()
    dense = tf.feature_column.input_layer(features, dense_feature)

    # stacked residual layer
    with tf.variable_scope('Residual_layers'):
        for i, unit in enumerate(params['hidden_units']):
            dense = residual_layer( dense, unit,
                                    dropout_rate = params['dropout_rate'],
                                    batch_norm = params['batch_norm'], mode = mode)
            add_layer_summary('residual_layer{}'.format(i), dense)

    with tf.variable_scope('output'):
        y = tf.layers.dense(dense, units=1)
        add_layer_summary( 'output', y )

    return y


build_estimator = build_estimator_helper(
    model_fn = {
        'census':model_fn
    },
    params = {
        'census':{'dropout_rate' : 0.2,
               'batch_norm' : True,
               'learning_rate' : 0.01,
               'hidden_units' : [10,5]
            }
    }
)