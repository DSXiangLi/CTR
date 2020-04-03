"""
Paper

Yuchin Juan，Yong Zhuang，Wei-Sheng Chin，Field-aware Factorization Machines for CTR Prediction

"""

from model.FFM.preprocess import build_features
import tensorflow as tf
import numpy as np
from config import *


def model_fn(features, labels, mode, params):
    """
    Field_aware factorication machine for 2 classes classification
    """
    feature_columns, field_dict = build_features()

    field_dim = len(np.unique(list(field_dict.values())))

    input = tf.feature_column.input_layer(features, feature_columns)

    input_dim = input.get_shape().as_list()[-1]

    with tf.variable_scope('linear'):
        init = tf.random_normal( shape = (input_dim,2) )
        w = tf.get_variable('w', dtype = tf.float32, initializer = init, validate_shape = False)
        b = tf.get_variable('b', shape = [2], dtype= tf.float32)

        linear_term = tf.add(tf.matmul(input,w), b)
        tf.summary.histogram( 'linear_term', linear_term )

    with tf.variable_scope('field_aware_interaction'):
        init = tf.truncated_normal(shape = (input_dim, field_dim, params['factor_dim']))
        v = tf.get_variable('v', dtype = tf.float32, initializer = init, validate_shape = False)

        interaction_term = tf.constant(0, dtype =tf.float32)
        # iterate over all the combination of features
        for i in range(input_dim):
            for j in range(i+1, input_dim):
                interaction_term += tf.multiply(
                    tf.reduce_mean(tf.multiply(v[i, field_dict[j],: ], v[j, field_dict[i],:])) ,
                    tf.multiply(input[:,i], input[:,j])
                )
        interaction_term = tf.reshape(interaction_term, [-1,1])
        tf.summary.histogram('interaction_term', interaction_term)

    with tf.variable_scope('output'):
        y = tf.math.add(interaction_term, linear_term)
        tf.summary.histogram( 'output', y )

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'predict_class': tf.argmax(tf.nn.softmax(y), axis=1),
            'prediction_prob': tf.nn.softmax(y)
        }

        return tf.estimator.EstimatorSpec(mode = tf.estimator.ModeKeys.PREDICT,
                                          predictions = predictions)

    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( labels=labels, logits=y ))

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate = params['learning_rate'])
        train_op = optimizer.minimize(cross_entropy,
                                     global_step = tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode, loss = cross_entropy, train_op = train_op)
    else:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels = labels,
                                            predictions = tf.argmax(tf.nn.softmax(y), axis=1)),
            'auc': tf.metrics.auc(labels = labels ,
                                  predictions = tf.nn.softmax(y)[:,1]),
            'pr': tf.metrics.auc(labels = labels,
                                 predictions = tf.nn.softmax(y)[:,1],
                                 curve = 'PR')
        }
        return tf.estimator.EstimatorSpec(mode, loss = cross_entropy, eval_metric_ops = eval_metric_ops)

def build_estimator(model_dir):

    run_config = tf.estimator.RunConfig(
        save_summary_steps=10,
        log_step_count_steps=10,
        keep_checkpoint_max = 3,
        save_checkpoints_steps =10
    )

    estimator = tf.estimator.Estimator(
        model_fn = model_fn,
        config = run_config,
        params = {
            'learning_rate' :0.001,
            'factor_dim': 5,
        },
        model_dir= model_dir
    )

    return estimator
