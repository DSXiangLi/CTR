import tensorflow as tf
from config import *

def parse_example_helper_csv(is_keras):
    def func(line):
        columns = tf.io.decode_csv( line, record_defaults=CSV_RECORD_DEFAULTS )

        features = dict( zip( FEATURE_NAME, columns ) )
        if is_keras:
            # it is so weird that keras to estimator need (,1) dimension for binary classification
            ## ToDo: figure out why
            target = tf.reshape( tf.cast( tf.equal( features.pop( TARGET ), '>50K' ), tf.int32 ), [-1, 1] )
        else:
            target = tf.reshape( tf.cast( tf.equal( features.pop( TARGET ), '>50K' ), tf.int32 ), [-1] )

        return features, target

    return func

def parse_example_helper_libsvm(is_keras):
    def func(line):
        pass
    return func

def input_fn(input_path, is_predict=0, is_keras=0, parse_csv=1):

    def func():
        dataset = tf.data.TextLineDataset( input_path ) \
            .skip( 1 ) \
            .batch( MODEL_PARAMS['batch_size'] )

        if parse_csv:
            parse_example = parse_example_helper_csv( is_keras )
        else:
            parse_example = parse_example_helper_libsvm(is_keras)
        dataset = dataset.map( parse_example, num_parallel_calls=8 )

        if is_predict==0:
            dataset = dataset \
                .shuffle(MODEL_PARAMS['buffer_size'] ) \
                .repeat(MODEL_PARAMS['num_epochs'] )

        return dataset
    return func

def tf_estimator_model(model_fn):
    def model_fn_helper(features, labels, mode, params):

        y = model_fn(features , labels, mode, params)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'predict_class': tf.argmax( tf.nn.softmax( y ), axis=1 ),
                'prediction_prob': tf.nn.softmax( y )
            }
            return tf.estimator.EstimatorSpec( mode=tf.estimator.ModeKeys.PREDICT,
                                               predictions=predictions )

        cross_entropy = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( labels=labels, logits=y ) )

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer( learning_rate=params['learning_rate'] )
            update_ops = tf.get_collection( tf.GraphKeys.UPDATE_OPS )
            with tf.control_dependencies( update_ops ):
                train_op = optimizer.minimize( cross_entropy,
                                               global_step=tf.train.get_global_step() )
            return tf.estimator.EstimatorSpec( mode, loss=cross_entropy, train_op=train_op )
        else:
            eval_metric_ops = {
                'accuracy': tf.metrics.accuracy( labels=labels,
                                                 predictions=tf.argmax( tf.nn.softmax( y ), axis=1 ) ),
                'auc': tf.metrics.auc( labels=labels,
                                       predictions=tf.nn.softmax( y )[:, 1] ),
                'pr': tf.metrics.auc( labels=labels,
                                      predictions=tf.nn.softmax( y )[:, 1],
                                      curve='PR' )
            }
            return tf.estimator.EstimatorSpec( mode, loss=cross_entropy, eval_metric_ops=eval_metric_ops )

    return model_fn_helper


def add_layer_summary(tag, value):
  tf.summary.scalar('{}/fraction_of_zero_values'.format(tag), tf.math.zero_fraction(value))
  tf.summary.histogram('{}/activation'.format(tag),  value)

