"""
paper

Weinan Zhang, Tianming Du, and Jun Wang. Deep learning over multi-field categorical data - - A case study on user response

"""

import tensorflow as tf
import numpy as np
from config import *
from utils import tf_estimator_model, add_layer_summary, build_estimator_helper
from model.FNN.preprocess import build_features

@tf_estimator_model
def model_fn(features, labels, mode, params):
    feature_columns= build_features()

    input = tf.feature_column.input_layer(features, feature_columns)

    with tf.variable_scope('init_fm_embedding'):
        # method1: load from checkpoint directly
        embeddings = tf.Variable( tf.contrib.framework.load_variable(
            './checkpoint/FM',
            'fm_interaction/v'
        ) )
        weight = tf.Variable( tf.contrib.framework.load_variable(
            './checkpoint/FM',
            'linear/w'
        ) )
        dense = tf.add(tf.matmul(input, embeddings), tf.matmul(input, weight))
        add_layer_summary('input', dense)

    with tf.variable_scope( 'Dense' ):
        for i, unit in enumerate( params['hidden_units'] ):
            dense = tf.layers.dense( dense, units=unit, activation='relu', name='dense{}'.format( i ) )
            dense = tf.layers.batch_normalization( dense, center=True, scale=True, trainable=True,
                                                   training=(mode == tf.estimator.ModeKeys.TRAIN) )
            dense = tf.layers.dropout( dense, rate=params['dropout_rate'],
                                       training=(mode == tf.estimator.ModeKeys.TRAIN) )
            add_layer_summary( dense.name, dense )

    with tf.variable_scope('output'):
        y = tf.layers.dense(dense, units= 1, name = 'output')
        tf.summary.histogram(y.name, y)

    return y

build_estimator = build_estimator_helper(
    mdoel_fn = {
        'dense':model_fn
    },
    params = {
        'dense': {
            'dropout_rate':0.2,
            'learning_rate': 0.002,
            'hidden_units':[24,12,1]
        }
    }
)


# check name of all the tensor in the checkpoint

if __name__ == '__main__':
    print ('checking name of all the tensor in the FNN pretrain checkpoint')
    from tensorflow.python.tools.inspect_checkpoint import  print_tensors_in_checkpoint_file
    latest_ckp = tf.train.latest_checkpoint('./checkpoint/FM')
    print_tensors_in_checkpoint_file( latest_ckp, all_tensors=True )
    print_tensors_in_checkpoint_file(latest_ckp, all_tensors=False, tensor_name='fm_interaction/v' )