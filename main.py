import argparse
import importlib
import pandas as pd
from config import *
from utils import *
import shutil

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def main(args):
    model = args.model
    build_estimator = getattr(importlib.import_module('model.{}.{}'.format(model, model)),
                        'build_estimator')

    model_dir = MODEL_DIR.format(model)

    if args.clear_model:
        try:
            shutil.rmtree(model_dir)
        except Exception as e:
            print('Error! {} occured at model cleanin'.format(e))
        else:
            print( '{} model cleaned'.format(model_dir) )

    estimator = build_estimator(model_dir)

    if args.type == 'train':
        early_stopping = tf.estimator.experimental.stop_if_no_decrease_hook(
            estimator,
            metric_name="loss",
            max_steps_without_decrease=100 * 40 )

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
    parser.add_argument('--clear_model', type=int, help='Whether to clear existing model', required=False, default=1)
    args = parser.parse_args()

    main(args)