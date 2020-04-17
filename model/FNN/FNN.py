"""
paper

Weinan Zhang, Tianming Du, and Jun Wang. Deep learning over multi-field categorical data - - A case study on user response

"""


from model.FNN.preprocess import build_features
import tensorflow as tf
import numpy as np
from config import *
from utils import tf_estimator_model, add_layer_summary

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
        add_layer_summary('fm_init_embeddings', embeddings)
        dense = tf.matmul(input, embeddings)
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

def build_estimator(model_dir):

    run_config = tf.estimator.RunConfig(
        save_summary_steps=50,
        log_step_count_steps=50,
        keep_checkpoint_max=3,
        save_checkpoints_steps=50
    )

    # load pretrained FM embedding layer to initialize the embedding layer in FNN
    estimator = tf.estimator.Estimator(
        model_fn = model_fn,
        config = run_config,
        params = {
            'learning_rate' :0.002,
            'hidden_units':[20,10],
            'dropout_rate':0.1,
        },
        model_dir= model_dir
    )

    return estimator


# check name of all the tensor in the checkpoint

if __name__ == '__main__':
    print ('checking name of all the tensor in the FNN pretrain checkpoint')
    from tensorflow.python.tools.inspect_checkpoint import  print_tensors_in_checkpoint_file
    latest_ckp = tf.train.latest_checkpoint('./checkpoint/FM')
    print_tensors_in_checkpoint_file( latest_ckp, all_tensors=True )
    print_tensors_in_checkpoint_file(latest_ckp, all_tensors=False, tensor_name='fm_interaction/v' )