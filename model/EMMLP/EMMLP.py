import tensorflow as tf
import numpy as np
from config import *
from model.EMMLP.preprocess import build_features

def add_layer_summary(tag, value):
  tf.summary.histogram('activation/{}'.format(tag), value)

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
                dense = tf.layers.dropout(dense, rate = params['dropout_rate'])

    with tf.variable_scope('output'):
        y = tf.layers.dense(dense, units=2, activation = 'relu', name = 'output')

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
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
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
        save_summary_steps= 10,
        log_step_count_steps= 10,
        keep_checkpoint_max = 3,
        save_checkpoints_steps = 10
    )

    # can choose to bucketize the numeric feature or concatenate directly with embedding
    numeric_handle = 'dense'
    model_dir = model_dir + '/'+ numeric_handle

    estimator = tf.estimator.Estimator(
        model_fn = model_fn,
        config = run_config,
        params = {
            'learning_rate' :0.01,
            'numeric_handle':numeric_handle,
            'hidden_units': [20,10],
            'embedding_dim': 5,
            'dropout_rate': 0.1
        },
        model_dir= model_dir
    )

    return estimator



