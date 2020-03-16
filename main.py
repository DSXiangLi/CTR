import argparse
import importlib
import pandas as pd
from config import *

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


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
            .batch( 10) \
            .map( parse_example, num_parallel_calls=8 )

        if is_predict==0:
            dataset = dataset \
                .shuffle(MODEL_PARAMS['buffer_size'] ) \
                .repeat(MODEL_PARAMS['num_epochs'] )

        return dataset
    return func


def main(args):
    model = args.model
    build_estimator = getattr(importlib.import_module('model.{}.{}'.format(model, model)),
                        'build_estimator')

    estimator = build_estimator(MODEL_DIR.format(model))

    if args.type == 'train':
        early_stopping = tf.estimator.experimental.stop_if_no_decrease_hook(
            estimator,
            metric_name="loss",
            max_steps_without_decrease=50 )

        train_spec = tf.estimator.TrainSpec( input_fn= input_fn(DATA_DIR.format('train'), is_keras =args.is_keras), hooks = [early_stopping])
        eval_spec = tf.estimator.EvalSpec( input_fn= input_fn(DATA_DIR.format('valid'), is_keras =args.is_keras,  is_predict=1 ))
        tf.estimator.train_and_evaluate( estimator, train_spec, eval_spec)

    if args.type =='predict':
        prediction = estimator.predict( input_fn=input_fn( DATA_DIR.format( 'valid' ), is_keras =args.is_keras,  is_predict=1 ) )
        predict_prob = pd.DataFrame({'predict_prob': [i['prediction_prob'][1] for i in prediction ]})
        predict_prob.to_csv('./result/prediction_{}.csv'.format(model))


if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument( '--model', type = str, help = 'which model to use[FM|FFM]',required=True )
    parser.add_argument( '--type', type = str, help = 'To train new model or load model to predit', required=False, default='train' )
    parser.add_argument( '--is_keras', type=int, help='Whether tf.estimator is built on keras', required=False, default=0 )
    args = parser.parse_args()

    main(args)