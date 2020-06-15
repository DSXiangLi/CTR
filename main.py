import argparse
import importlib
import shutil
import pandas as pd

from config import CONFIG
from utils import *

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def main(args):
    model = args.model
    config = CONFIG( model_name = model, data_name = args.dataset )

    # clear Existing Model
    if args.clear_model:
        try:
            shutil.rmtree(config.checkpoint_dir)
        except Exception as e:
            print('Error! {} occured at model cleaning'.format(e))
        else:
            print( '{} model cleaned'.format(config.checkpoint_dir) )

    # build estimator
    build_estimator = getattr(importlib.import_module('model.{}.{}'.format(model, model)),
                             'build_estimator')
    estimator = build_estimator(config)

    # train or predict
    if args.step == 'train':
        early_stopping = tf.estimator.experimental.stop_if_no_decrease_hook(
            estimator,
            metric_name="loss",
            max_steps_without_decrease= 20 * 100 )

        train_spec = tf.estimator.TrainSpec( input_fn = input_fn( step = 'train',
                                             is_predict = 0,
                                            config = config), hooks = [early_stopping])

        eval_spec = tf.estimator.EvalSpec( input_fn = input_fn( step ='valid',
                                           is_predict = 1,
                                           config = config ),
                                           steps = 200,
                                           throttle_secs = 60)

        tf.estimator.train_and_evaluate( estimator, train_spec, eval_spec)

    if args.step =='predict':
        prediction = estimator.predict( input_fn = input_fn( step='valid',
                                        is_predict = 1,
                                        config = config) )

        predict_prob = pd.DataFrame({'predict_prob': [i['prediction_prob'][1] for i in prediction ]})
        predict_prob.to_csv('./result/prediction_{}.csv'.format(model))


if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument( '--model', type = str, help = 'which model to use[FM|FFM]', required=True )
    parser.add_argument( '--step', type = str, help = 'Train or Predict', required=False, default='train' )
    parser.add_argument( '--clear_model', type=int, help= 'Whether to clear existing model', required=False, default=1)
    parser.add_argument( '--dataset', type=str, help= 'which dataset to use [frappe, census, amazon]',
                         required=False, default='dense')
    args = parser.parse_args()

    main(args)
