"""
paper

Xiangnan He, Tat-Seng Chua,2017,  Neural Factorization Machines for Sparse Predictive Analytics
"""


import tensorflow as tf
import numpy as np
from config import *
from model.NFM.preprocess import build_features
from utils import tf_estimator_model, add_layer_summary, build_estimator_helper

@tf_estimator_model
def model_fn(features, labels, mode, params):
    dense_feature = build_features()
    dense = tf.feature_column.input_layer(features, dense_feature) # lz linear concat of embedding

    feature_size = len( dense_feature )
    embedding_size = dense_feature[0].variable_shape.as_list()[-1]
    embedding_matrix = tf.reshape( dense, [-1, feature_size, embedding_size] )  # batch * feature_size *emb_size

    with tf.variable_scope('BI_Pooling'):
        sum_square = tf.pow(tf.reduce_sum(embedding_matrix, axis=1), 2)
        square_sum = tf.reduce_sum(tf.pow(embedding_matrix, 2), axis=1)
        dense = tf.subtract(sum_square, square_sum)
        add_layer_summary( dense.name, dense )

    with tf.variable_scope('Dense'):
        for i, unit in enumerate( params['hidden_units'] ):
            dense = tf.layers.dense( dense, units=unit, activation='relu', name='dense{}'.format( i ) )
            dense = tf.layers.batch_normalization( dense, center=True, scale=True, trainable=True,
                                                   training=(mode == tf.estimator.ModeKeys.TRAIN) )
            dense = tf.layers.dropout( dense, rate=params['dropout_rate'],
                                       training=(mode == tf.estimator.ModeKeys.TRAIN) )
            add_layer_summary( dense.name, dense)

    with tf.variable_scope('output'):
        y = tf.layers.dense(dense, units=1, name = 'output')
        add_layer_summary( 'output', y )

    return y


build_estimator = build_estimator_helper(
    {'dense':model_fn},
     params = {'dropout_rate':0.2,
               'learning_rate' :0.002,
               'hidden_units':[5,5]
            }
)