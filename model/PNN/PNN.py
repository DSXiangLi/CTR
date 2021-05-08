"""
paper

Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction,2016 IEEE

"""
import tensorflow as tf
import numpy as np
from const import *
from model.PNN.preprocess import build_features
from utils import tf_estimator_model, add_layer_summary, build_estimator_helper

@tf_estimator_model
def model_fn(features, labels, mode, params):
    dense_feature= build_features()
    dense = tf.feature_column.input_layer(features, dense_feature) # lz linear concat of embedding

    feature_size = len( dense_feature )
    embedding_size = dense_feature[0].variable_shape.as_list()[-1]
    embedding_matrix = tf.reshape( dense, [-1, feature_size, embedding_size] )  # batch * feature_size *emb_size

    with tf.variable_scope('IPNN'):
        # use matrix multiplication to perform inner product of embedding
        inner_product = tf.matmul(embedding_matrix, tf.transpose(embedding_matrix, perm=[0,2,1])) # batch * feature_size * feature_size
        inner_product = tf.reshape(inner_product, [-1, feature_size * feature_size ])# batch * (feature_size * feature_size)
        add_layer_summary(inner_product.name, inner_product)

    with tf.variable_scope('OPNN'):
        outer_collection = []
        for i in range(feature_size):
            for j in range(i+1, feature_size):
                vi = tf.gather(embedding_matrix, indices = i, axis=1, batch_dims=0, name = 'vi') # batch * embedding_size
                vj = tf.gather(embedding_matrix, indices = j, axis=1, batch_dims= 0, name='vj') # batch * embedding_size
                outer_collection.append(tf.reshape(tf.einsum('ai,aj->aij',vi,vj), [-1, embedding_size * embedding_size])) # batch * (emb * emb)

        outer_product = tf.concat(outer_collection, axis=1)
        add_layer_summary( outer_product.name, outer_product )

    with tf.variable_scope('fc1'):
        if params['model_type'] == 'IPNN':
            dense = tf.concat([dense, inner_product], axis=1)
        elif params['model_type'] == 'OPNN':
            dense = tf.concat([dense, outer_product], axis=1)
        elif params['model_type'] == 'PNN':
            dense = tf.concat([dense, inner_product, outer_product], axis=1)
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
    model_fn = {
        'census':model_fn
    },
    params = {
        'census': {
            'model_type':'IPNN',  # support IPNN/OPNN/PNN
            'dropout_rate':0.2,
            'learning_rate': 0.01,
            'hidden_units':[24,12,1]
        }
    }
)
