"""
paper
Jianxun Lian， 2018， xDeepFM - Combining Explicit and Implicit Feature Interactions for Recommender Systems

"""

import tensorflow as tf
import numpy as np
from const import *
from model.DeepFM.preprocess import build_features
from utils import tf_estimator_model, add_layer_summary, build_estimator_helper
from layers import stack_dense_layer, sparse_embedding,sparse_linear


def cross_op(xk, x0, layer_size_prev, layer_size_curr, layer, emb_size, field_size):
    # Hamard product: ( batch * D * HK-1 * 1) * (batch * D * 1* H0) -> batch * D * HK-1 * H0
    zk = tf.matmul( tf.expand_dims(tf.transpose(xk, perm = (0, 2, 1)), 3),
                    tf.expand_dims(tf.transpose(x0, perm = (0, 2, 1)), 2))

    zk = tf.reshape(zk, [-1, emb_size, field_size * layer_size_prev]) # batch * D * HK-1 * H0 -> batch * D * (HK-1 * H0)
    add_layer_summary('zk_{}'.format(layer), zk)

    # Convolution with channel = HK: (batch * D * (HK-1*H0)) * ((HK-1*H0) * HK)-> batch * D * HK
    kernel = tf.get_variable(name = 'kernel{}'.format(layer),
                             shape = (field_size * layer_size_prev, layer_size_curr))
    xkk = tf.matmul(zk, kernel)
    xkk = tf.transpose(xkk, perm = [0,2,1]) # batch * HK * D
    add_layer_summary( 'Xk_{}'.format(layer), xkk )
    return xkk


def cin_layer(x0, cin_layer_size, emb_size, field_size):
    cin_output_list = []

    cin_layer_size.insert(0, field_size) # insert field dimension for input
    with tf.variable_scope('Cin_component'):
        xk = x0
        for layer in range(1, len(cin_layer_size)):
            with tf.variable_scope('Cin_layer{}'.format(layer)):
                # Do cross
                xk = cross_op(xk, x0, cin_layer_size[layer-1], cin_layer_size[layer],
                              layer, emb_size, field_size ) # batch * HK * D
                # sum pooling on dimension axis
                cin_output_list.append(tf.reduce_sum(xk, 2)) # batch * HK

    return tf.concat(cin_output_list, axis=1)


@tf_estimator_model
def model_fn_dense(features, labels, mode, params):
    dense_feature, sparse_feature = build_features()
    dense_input = tf.feature_column.input_layer(features, dense_feature)
    sparse_input = tf.feature_column.input_layer(features, sparse_feature)

    # Linear part
    with tf.variable_scope('Linear_component'):
        linear_output = tf.layers.dense( sparse_input, units=1 )
        add_layer_summary( 'linear_output', linear_output )

    # Deep part
    dense_output = stack_dense_layer( dense_input, params['hidden_units'],
                               params['dropout_rate'], params['batch_norm'],
                               mode, add_summary=True )
    # CIN part
    emb_size = dense_feature[0].variable_shape.as_list()[-1]
    field_size = len(dense_feature)
    embedding_matrix = tf.reshape(dense_input, [-1, field_size, emb_size]) # batch * field_size * emb_size
    add_layer_summary('embedding_matrix', embedding_matrix)

    cin_output = cin_layer(embedding_matrix, params['cin_layer_size'], emb_size, field_size)

    with tf.variable_scope('output'):
        y = tf.concat([dense_output, cin_output,linear_output], axis=1)
        y = tf.layers.dense(y, units= 1)
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
    feat_ids = tf.reshape(features['feat_ids'], shape = [-1, field_size]) # batch * field_size
    feat_vals = tf.reshape(features['feat_vals'], shape = [-1, field_size]) # batch * field_size

    # extract embedding
    with tf.variable_scope('extract_embedding'):
        embedding_matrix = sparse_embedding( feature_size, embedding_size, field_size,
                                             feat_ids, feat_vals, add_summary =True) # (batch, field_size, embedding_size)
        dense_input = tf.reshape( embedding_matrix,
                                  [-1, field_size * embedding_size] )  # (batch, field_size * embedding_size)

    # linear part
    linear_output = sparse_linear( feature_size, feat_ids, feat_vals, add_summary = True )

    # Deep part
    dense_output = stack_dense_layer(dense_input, params['hidden_units'],
                                   params['dropout_rate'], params['batch_norm'],
                                   mode, add_summary=True )
    # CIN part
    cin_output = cin_layer(embedding_matrix, params['cin_layer_size'], embedding_size, field_size )

    # concat and output
    with tf.variable_scope('output'):
        y = tf.concat( [dense_output, cin_output, linear_output], axis=1 )
        y = tf.layers.dense( y, units=1 )
        add_layer_summary( 'output', y )

    return y


build_estimator = build_estimator_helper(
    model_fn = {
        'census' : model_fn_dense,
        'frappe': model_fn_sparse
    },
    params = {
         'census': {
            'dropout_rate': 0.2,
            'learning_rate' : 0.01,
            'hidden_units': [20,10],
            'batch_norm': True,
            'cin_layer_size': [8,4,4]
            },
        'frappe': {
            'dropout_rate': 0.2,
            'learning_rate': 0.01,
            'hidden_units': [128, 64, 32],
            'batch_norm': True,
            'cin_layer_size': [32,16,8],
            'data_params': FRAPPE_PARAMS
        }
    }
)

