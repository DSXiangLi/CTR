"""
paper

Huifeng Guo et all. "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction," In IJCAI,2017.

"""

import tensorflow as tf
import numpy as np
from const import *
from model.DeepFM.preprocess import build_features
from utils import tf_estimator_model, add_layer_summary, build_estimator_helper


@tf_estimator_model
def model_fn_dense(features, labels, mode, params):
    dense_feature, sparse_feature = build_features()
    dense = tf.feature_column.input_layer(features, dense_feature)
    sparse = tf.feature_column.input_layer(features, sparse_feature)

    with tf.variable_scope('FM_component'):
        with tf.variable_scope( 'Linear' ):
            linear_output = tf.layers.dense(sparse, units=1)
            add_layer_summary( 'linear_output', linear_output )

        with tf.variable_scope('second_order'):
            # reshape (batch_size, n_feature * emb_size) -> (batch_size, n_feature, emb_size)
            emb_size = dense_feature[0].variable_shape.as_list()[-1]# all feature has same emb dimension
            embedding_matrix = tf.reshape(dense, (-1, len(dense_feature), emb_size))
            add_layer_summary( 'embedding_matrix', embedding_matrix )
            # Compared to FM embedding here is flatten(x * v) not v
            sum_square = tf.pow( tf.reduce_sum( embedding_matrix, axis=1 ), 2 )
            square_sum = tf.reduce_sum( tf.pow(embedding_matrix,2), axis=1 )

            fm_output = tf.reduce_sum(tf.subtract( sum_square, square_sum) * 0.5, axis=1, keepdims=True)
            add_layer_summary('fm_output', fm_output)

    with tf.variable_scope('Deep_component'):
        for i, unit in enumerate(params['hidden_units']):
            dense = tf.layers.dense(dense, units = unit, activation ='relu', name = 'dense{}'.format(i))
            dense = tf.layers.batch_normalization(dense, center=True, scale = True, trainable=True,
                                                  training=(mode ==tf.estimator.ModeKeys.TRAIN))
            dense = tf.layers.dropout( dense, rate=params['dropout_rate'], training = (mode==tf.estimator.ModeKeys.TRAIN))
            add_layer_summary( dense.name, dense )

    with tf.variable_scope('output'):
        y = dense + fm_output + linear_output
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

    with tf.variable_scope('FM_component'):
        bias = tf.get_variable(shape =[1], name = 'linear_bias', initializer = tf.glorot_uniform_initializer())
        weight = tf.get_variable(shape = [feature_size], name = 'linear_weight', initializer = tf.truncated_normal_initializer())
        v = tf.get_variable(shape = [feature_size, embedding_size], name = 'embedding_weight', initializer = tf.truncated_normal_initializer())

        with tf.variable_scope( 'Linear' ):
            # batch_size * feature_size -> batch_size * field_size  -> batch_size * 1
            linear_output = tf.reduce_sum(tf.multiply(tf.nn.embedding_lookup(weight, feat_ids), feat_vals) , axis=1, keepdims=True)
            linear_output = tf.add(linear_output,bias) # batch * 1
            add_layer_summary( 'linear_output', linear_output )

        with tf.variable_scope( 'second_order' ):
            embedding_matrix = tf.nn.embedding_lookup(v, feat_ids) # batch * field_size * emb_size
            embedding_matrix = tf.multiply(embedding_matrix, tf.reshape(feat_vals, [-1, field_size, 1])) # batch * field_size * emb_size
            add_layer_summary( 'embedding_matrix', embedding_matrix )

            sum_square = tf.pow( tf.reduce_sum( embedding_matrix, axis=1 ), 2 )
            square_sum = tf.reduce_sum( tf.pow(embedding_matrix,2), axis=1 )

            fm_output = tf.reduce_sum(tf.subtract( sum_square, square_sum) * 0.5, axis=1, keepdims=True) # batch * 1
            add_layer_summary('fm_output', fm_output)

    with tf.variable_scope('Deep_component'):
        dense = tf.reshape(embedding_matrix, shape = [-1, field_size * embedding_size], name = 'dense_embedding') # flatten embedding matrix
        for i, unit in enumerate(params['hidden_units']):
            dense = tf.layers.dense(dense, units = unit, activation ='relu', name = 'dense{}'.format(i))
            dense = tf.layers.batch_normalization(dense, center=True, scale = True, trainable=True,
                                                  training=(mode == tf.estimator.ModeKeys.TRAIN))
            dense = tf.layers.dropout( dense, rate=params['dropout_rate'], training = (mode==tf.estimator.ModeKeys.TRAIN))
            add_layer_summary( dense.name, dense )

    with tf.variable_scope( 'output' ):
        y = dense + fm_output+ linear_output # batch * 1
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
            'learning_rate' :0.01,
            'hidden_units':[20,10,1]
            },
        'frappe': {
            'dropout_rate':0.2,
            'learning_rate':0.01,
            'hidden_units':[128,64,1],
            'data_params': FRAPPE_PARAMS
        }
    }
)

