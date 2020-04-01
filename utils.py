import tensorflow as tf
from config import *

def parse_example_helper(is_keras):
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

def input_fn(input_path, is_predict=0, is_keras=0):

    def func():
        parse_example = parse_example_helper( is_keras )

        dataset = tf.data.TextLineDataset(input_path) \
            .skip(1) \
            .batch(MODEL_PARAMS['batch_size']) \
            .map( parse_example, num_parallel_calls=8 )

        if is_predict==0:
            dataset = dataset \
                .shuffle(MODEL_PARAMS['buffer_size'] ) \
                .repeat(MODEL_PARAMS['num_epochs'] )

        return dataset
    return func