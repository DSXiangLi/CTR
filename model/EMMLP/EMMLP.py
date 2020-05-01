import tensorflow as tf
import numpy as np
from config import *
from model.EMMLP.preprocess import build_features
from utils import tf_estimator_model, add_layer_summary, build_estimator_helper


@tf_estimator_model
def model_fn(features, labels, mode, params):
    sparse_columns, dense_columns = build_features(params['numeric_handle'])

    with tf.variable_scope('EmbeddingInput'):
        embedding_input = []
        for f_sparse in sparse_columns:
            sparse_input = tf.feature_column.input_layer(features, f_sparse)

            input_dim = sparse_input.get_shape().as_list()[-1]

            init = tf.random_normal(shape = [input_dim, params['embedding_dim']])

            weight = tf.get_variable('w_{}'.format(f_sparse.name), dtype = tf.float32, initializer = init)

            add_layer_summary(weight.name, weight)

            embedding_input.append( tf.matmul(sparse_input, weight) )

        dense = tf.concat(embedding_input, axis=1, name = 'embedding_concat')
        add_layer_summary( dense.name, dense )

        # if treat numeric feature as dense feature, then concatenate with embedding. else concatenate wtih sparse input
        if params['numeric_handle'] == 'dense':
            numeric_input = tf.feature_column.input_layer(features, dense_columns)

            numeric_input = tf.layers.batch_normalization(numeric_input, center = True, scale = True, trainable =True,
                                                          training = (mode == tf.estimator.ModeKeys.TRAIN))
            add_layer_summary( numeric_input.name, numeric_input )
            dense = tf.concat([dense, numeric_input], axis = 1, name ='numeric_concat')
            add_layer_summary(dense.name, dense)

    with tf.variable_scope('MLP'):
        for i, unit in enumerate(params['hidden_units']):
            dense = tf.layers.dense(dense, units = unit, activation = 'relu', name = 'Dense_{}'.format(i))
            if mode == tf.estimator.ModeKeys.TRAIN:
                add_layer_summary(dense.name, dense)
                dense = tf.layers.dropout(dense, rate = params['dropout_rate'], training = (mode==tf.estimator.ModeKeys.TRAIN))

    with tf.variable_scope('output'):
        y = tf.layers.dense(dense, units=1, name = 'output')

    return y


build_estimator = build_estimator_helper(
     model_fn = {
         'dense':model_fn
     },
     params = {
         'dense':
             {'learning_rate' :0.002,
            'numeric_handle':'dense', # dense or bucketize are supported
            'hidden_units': [20,10],
            'embedding_dim': 4,
            'dropout_rate': 0.1
            }
     }
)

