"""
paper

Huifeng Guo et all. "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction," In IJCAI,2017.

"""
import tensorflow as tf
import numpy as np
from config import *
from model.DeepFM.preprocess import build_features
from utils import tf_estimator_model, add_layer_summary

@tf_estimator_model
def model_fn(features, labels, mode, params):
    dense_feature, sparse_feature = build_features()
    dense = tf.feature_column.input_layer(features, dense_feature)
    sparse = tf.feature_column.input_layer(features, sparse_feature)

    with tf.variable_scope('FM_component'):
        with tf.variable_scope( 'Linear' ):
            bias = tf.random_normal(shape = (1,))
            init = tf.random_normal( shape=[sparse.get_shape().as_list()[-1], 1] )
            w = tf.get_variable( 'linear_weight', dtype=tf.float32, initializer=init )
            linear_output = bias + tf.matmul( sparse, w )
            add_layer_summary( 'linear_output', linear_output )

        with tf.variable_scope('second_order'):
            # reshape (batch_size, n_feature * emb_size) -> (batch_size, n_feature, emb_size)
            emb_size = dense_feature[0].variable_shape.as_list()[0] # all feature has same emb dimension
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

def build_estimator(model_dir):

    run_config = tf.estimator.RunConfig(
        save_summary_steps=50,
        log_step_count_steps=50,
        keep_checkpoint_max = 3,
        save_checkpoints_steps =50
    )

    estimator = tf.estimator.Estimator(
        model_fn = model_fn,
        config = run_config,
        params = {
            'dropout_rate':0.2,
            'learning_rate' :0.001,
            'hidden_units':[20,10,1]
        },
        model_dir= model_dir
    )

    return estimator
