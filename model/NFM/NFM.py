"""
paper

Xiangnan He, Tat-Seng Chua,2017,  Neural Factorization Machines for Sparse Predictive Analytics
"""


import tensorflow as tf
import numpy as np
from const import *
from model.NFM.preprocess import build_features
from utils import tf_estimator_model, add_layer_summary, build_estimator_helper
from layers import sparse_embedding, sparse_linear, stack_dense_layer

@tf_estimator_model
def model_fn_dense(features, labels, mode, params):
    dense_feature, sparse_feature = build_features()
    dense = tf.feature_column.input_layer(features, dense_feature)
    sparse = tf.feature_column.input_layer(features, sparse_feature)

    field_size = len( dense_feature )
    embedding_size = dense_feature[0].variable_shape.as_list()[-1]
    embedding_matrix = tf.reshape( dense, [-1, field_size, embedding_size] )  # batch * field_size *emb_size

    with tf.variable_scope('Linear_output'):
        linear_output = tf.layers.dense( sparse, units=1 )
        add_layer_summary( 'linear_output', linear_output )

    with tf.variable_scope('BI_Pooling'):
        sum_square = tf.pow(tf.reduce_sum(embedding_matrix, axis=1), 2)
        square_sum = tf.reduce_sum(tf.pow(embedding_matrix, 2), axis=1)
        dense = tf.subtract(sum_square, square_sum)
        add_layer_summary( dense.name, dense )

    dense = stack_dense_layer(dense, params['hidden_units'],
                              dropout_rate = params['dropout_rate'], batch_norm = params['batch_norm'],
                              mode = mode, add_summary = True)

    with tf.variable_scope('output'):
        y = linear_output + dense
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
    embedding_matrix = sparse_embedding(feature_size, embedding_size, field_size,
                                        feat_ids, feat_vals, add_summary=True)

    # linear output
    linear_output = sparse_linear(feature_size, feat_ids, feat_vals, add_summary= True)

    with tf.variable_scope('BI_Pooling'):
        sum_square = tf.pow(tf.reduce_sum(embedding_matrix, axis=1),2)
        square_sum = tf.reduce_sum(tf.pow(embedding_matrix,2), axis=1)
        dense = tf.subtract(sum_square, square_sum)
        add_layer_summary( dense.name, dense )

    # fully connected stacked dense layers
    dense = stack_dense_layer( dense, params['hidden_units'],
                               dropout_rate=params['dropout_rate'], batch_norm=params['batch_norm'],
                               mode = mode, add_summary = True)

    with tf.variable_scope( 'output' ):
        y = linear_output + dense
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
            'batch_norm': True,
            'learning_rate' :0.01,
            'hidden_units':[20,10,1]
            },
        'frappe': {
            'dropout_rate': 0.2,
            'batch_norm': True,
            'learning_rate': 0.01,
            'hidden_units': [128,64,1],
            'data_params': FRAPPE_PARAMS

        }
    }
)