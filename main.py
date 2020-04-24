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

    # clear Existing Model
    if args.clear_model:
        try:
            shutil.rmtree(model_dir)
        except Exception as e:
            print('Error! {} occured at model cleanin'.format(e))
        else:
            print( '{} model cleaned'.format(model_dir) )

    # build estimator
    estimator = build_estimator(model_dir, input_type = args.input_type)

    # train or predict
    if args.step == 'train':
        early_stopping = tf.estimator.experimental.stop_if_no_decrease_hook(
            estimator,
            metric_name="loss",
            max_steps_without_decrease= 100 * 100 )

        train_spec = tf.estimator.TrainSpec( input_fn = input_fn( data_dir( 'train', args.input_type),
                                             expand_dimension = args.expand_dimension,
                                             is_predict = 0,
                                             input_type = args.input_type), hooks = [early_stopping])

        eval_spec = tf.estimator.EvalSpec( input_fn = input_fn( data_dir( 'valid', args.input_type),
                                           expand_dimension = args.expand_dimension,
                                           is_predict = 1,
                                           input_type = args.input_type),
                                           steps = 200,
                                           throttle_secs = 60)

        tf.estimator.train_and_evaluate( estimator, train_spec, eval_spec)

    if args.step =='predict':
        prediction = estimator.predict( input_fn = input_fn( data_dir( 'valid', args.input_type),
                                        expand_dimension = args.expand_dimension,
                                        is_predict = 1,
                                        input_type = args.input_type) )

        predict_prob = pd.DataFrame({'predict_prob': [i['prediction_prob'][1] for i in prediction ]})
        predict_prob.to_csv('./result/prediction_{}.csv'.format(model))


if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument( '--model', type = str, help = 'which model to use[FM|FFM]', required=True )
    parser.add_argument( '--step', type = str, help = 'Train or Predict', required=False, default='train' )
    parser.add_argument( '--expand_dimension', type=int, help='whether to expand label dimension by 1',
                         required=False, default=1 )
    parser.add_argument( '--clear_model', type=int, help='Whether to clear existing model', required=False, default=1)
    parser.add_argument( '--input_type', type=str, help='Use dense input[adult data in csv] or sparse input[crito data in libsvm]',
                         required=False, default='dense')
    args = parser.parse_args()

    main(args)