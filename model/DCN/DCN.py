"""

paper

Gang Fu,Mingliang Wang, 2017, Deep & Cross Network for Ad Click Predictions

"""

import tensorflow as tf
import numpy as np
from const import *
from model.DCN.preprocess import build_features
from utils import tf_estimator_model, add_layer_summary, build_estimator_helper
from layers import stack_dense_layer, sparse_embedding

def cross_op_raw(xl, x0, weight, feature_size):
    # original multiply order: (x0 * xl) * w
    # (batch,feature_size) - > (batch, feature_size * feature_size)
    outer_product = tf.matmul(tf.reshape(x0, [-1, feature_size,1]),
                              tf.reshape(xl, [-1, 1, feature_size])
                              )
    # (batch,feature_size*feature_size) ->(batch, feature_size)
    interaction = tf.tensordot(outer_product, weight, axes=1)
    return interaction

def cross_op_better(xl, x0, weight, feature_size):
    # change multiply order x0 * xl * w -> x0 * (xl * w)
    # (batch, 1, feature_size) * (featuure_size) -> (batch,1)
    transform = tf.tensordot( tf.reshape( xl, [-1, 1, feature_size] ), weight, axes=1 )
    # (batch, feature_size) * (batch, 1) -> (batch, feature_size)
    interaction = tf.multiply( x0, transform )

    return interaction

def cross_layer(x0, cross_layers, cross_op = 'better'):
    xl = x0
    if cross_op == 'better':
        cross_func = cross_op_better
    else:
        cross_func = cross_op_raw

    with tf.variable_scope( 'cross_layer' ):
        feature_size = x0.get_shape().as_list()[-1]  # feature_size = n_feature * embedding_size
        for i in range( cross_layers):
            weight = tf.get_variable( shape=[feature_size],
                                      initializer=tf.truncated_normal_initializer(), name='cross_weight{}'.format( i ) )
            bias = tf.get_variable( shape=[feature_size],
                                    initializer=tf.truncated_normal_initializer(), name='cross_bias{}'.format( i ) )

            interaction = cross_func(xl, x0, weight, feature_size)

            xl = interaction + bias + xl  # add back previous layer  -> (batch, feature_size)
            add_layer_summary( 'cross_{}'.format( i ), xl )
    return xl

@tf_estimator_model
def model_fn_dense(features, labels, mode, params):
    dense_feature = build_features()
    dense_input = tf.feature_column.input_layer(features, dense_feature)

    # deep part
    dense = stack_dense_layer(dense_input, params['hidden_units'],
                              params['dropout_rate'], params['batch_norm'],
                              mode, add_summary = True)

    # cross part
    xl = cross_layer(dense_input, params['cross_layers'], params['cross_op'])

    with tf.variable_scope('stack'):
        x_stack = tf.concat( [dense, xl], axis=1 )

    with tf.variable_scope('output'):
        y = tf.layers.dense(x_stack, units =1)
        add_layer_summary( 'output', y )

    return y


@tf_estimator_model
def model_fn_sparse(features, labels, mode, params):
    # hyper parameter
    data_params = params['data_params']
    field_size = data_params['field_size']
    feature_size = data_params['feature_size']
    embedding_size = data_params['embedding_size']

    # extract feature
    feat_ids = tf.reshape(features['feat_ids'], shape = [-1, field_size]) # (batch, field_size)
    feat_vals = tf.reshape(features['feat_vals'], shape = [-1, field_size]) # (batch, field_size)

    # extract embedding
    with tf.variable_scope('extract_embedding'):
        embedding_matrix = sparse_embedding( feature_size, embedding_size, field_size,
                                             feat_ids, feat_vals, add_summary =True) # (batch, field_size, embedding_size)
        dense_input = tf.reshape( embedding_matrix,
                                  [-1, field_size * embedding_size] )  # (batch, field_size * embedding_size)
    # deep part
    dense = stack_dense_layer(dense_input, params['hidden_units'],
                              params['dropout_rate'], params['batch_norm'],
                              mode, add_summary = True)

    # cross part
    xl = cross_layer(dense_input, params['cross_layers'])

    with tf.variable_scope('stack'):
        x_stack = tf.concat( [dense, xl], axis=1 )

    with tf.variable_scope('output'):
        y = tf.layers.dense(x_stack, units =1)
        add_layer_summary( 'output', y )

    return y

build_estimator = build_estimator_helper(
    model_fn = {
        'census': model_fn_dense,
        'frappe': model_fn_sparse
    },
     params = {
         'census':{
               'dropout_rate' : 0.2,
               'batch_norm' : True,
               'learning_rate' : 0.01,
               'hidden_units' : [10,5],
               'cross_layers' : 3,
               'cross_op':'raw'
         },
         'frappe':{
               'dropout_rate' : 0.2,
               'batch_norm' : True,
               'learning_rate' : 0.01,
               'hidden_units' : [128, 64],
               'cross_layers' : 3,
               'cross_op':'better',
               'data_params': FRAPPE_PARAMS
         }
   }
)