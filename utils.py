import tensorflow as tf
from config import *

def parse_example_helper_csv(expand_dimension):
    def func(line):
        columns = tf.io.decode_csv( line, record_defaults=CSV_RECORD_DEFAULTS )

        features = dict( zip( FEATURE_NAME, columns ) )
        if expand_dimension:
            # it is so weird that keras to estimator need (,1) dimension for binary classification
            ## ToDo: figure out why
            target = tf.reshape( tf.cast( tf.equal( features.pop( TARGET ), '>50K' ), tf.float32 ), [-1, 1] )
        else:
            target = tf.reshape( tf.cast( tf.equal( features.pop( TARGET ), '>50K' ), tf.int32 ), [-1] )

        return features, target

    return func

def parse_example_helper_libsvm(expand_dimension):
    def func(line):
        pass
    return func

def input_fn(input_path, is_predict=0, expand_dimension=0, parse_csv=1):

    def func():
        dataset = tf.data.TextLineDataset( input_path ) \
            .skip( 1 ) \
            .batch( MODEL_PARAMS['batch_size'] )

        if parse_csv:
            parse_example = parse_example_helper_csv( expand_dimension )
        else:
            parse_example = parse_example_helper_libsvm(expand_dimension)
        dataset = dataset.map( parse_example, num_parallel_calls=8 )

        if is_predict==0:
            dataset = dataset \
                .shuffle(MODEL_PARAMS['buffer_size'] ) \
                .repeat(MODEL_PARAMS['num_epochs'] )

        return dataset
    return func


def add_layer_summary(tag, value):
  tf.summary.scalar('{}/fraction_of_zero_values'.format(tag), tf.math.zero_fraction(value))
  tf.summary.histogram('{}/activation'.format(tag),  value)


def tf_estimator_model(model_fn):
    def model_fn_helper(features, labels, mode, params):

        y = model_fn(features , labels, mode, params)

        add_layer_summary('label_mean', labels)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'prediction_prob': tf.sigmoid( y )
            }
            return tf.estimator.EstimatorSpec( mode=tf.estimator.ModeKeys.PREDICT,
                                               predictions=predictions )

        cross_entropy = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( labels=labels, logits=y ) )

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdagradOptimizer( learning_rate=params['learning_rate'] )
            update_ops = tf.get_collection( tf.GraphKeys.UPDATE_OPS )
            with tf.control_dependencies( update_ops ):
                train_op = optimizer.minimize( cross_entropy,
                                               global_step=tf.train.get_global_step() )
            return tf.estimator.EstimatorSpec( mode, loss=cross_entropy, train_op=train_op )
        else:
            eval_metric_ops = {
                'accuracy': tf.metrics.accuracy( labels=labels,
                                                 predictions=tf.to_float(tf.greater_equal(tf.sigmoid(y),0.5))  ),
                'auc': tf.metrics.auc( labels=labels,
                                       predictions=tf.sigmoid( y )),
                'pr': tf.metrics.auc( labels=labels,
                                      predictions=tf.sigmoid( y ),
                                      curve='PR' )
            }
            return tf.estimator.EstimatorSpec( mode, loss=cross_entropy, eval_metric_ops=eval_metric_ops )

    return model_fn_helper

