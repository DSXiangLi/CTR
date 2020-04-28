import tensorflow as tf
from config import *

def parse_example_helper_csv(line):
    columns = tf.io.decode_csv( [line], record_defaults=CSV_RECORD_DEFAULTS )

    features = dict( zip( FEATURE_NAME, columns ) )

    target = tf.reshape( tf.cast( tf.equal( features.pop( TARGET ), '>50K' ), tf.float32 ), [-1] )

    return features, target


def parse_example_helper_libsvm(line):
    # '0 1:0 2:0.053068 3:0.5 4:0.1 5:0.113437 6:0.874'
    columns = tf.string_split([line], ' ')

    target = tf.string_to_number(columns.values[0], out_type = tf.float32)
    target = tf.reshape(tf.cast(target, tf.int32), [-1])

    splits = tf.string_split(columns.values[1:], ':')
    id_vals = tf.reshape(splits.values, splits.dense_shape )

    feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits =2, axis=1)
    feat_ids = tf.string_to_number(feat_ids , out_type = tf.int32)
    feat_vals = tf.string_to_number(feat_vals, out_type = tf.float32)

    return {'feat_ids': feat_ids, 'feat_vals': feat_vals}, target



def input_fn(input_path, is_predict, input_type):
    def func():
        if input_type == 'dense':
            # currently dense default to adult training set with csv format
            parse_example = parse_example_helper_csv
        elif input_type == 'sparse':
            # currently sparse default to criteo training set with libsvm format
            parse_example = parse_example_helper_libsvm
        else:
            raise Exception('Only dense and sparse are supported now')

        dataset = tf.data.TextLineDataset( input_path ) \
            .skip( 1 ) \
            .map( parse_example, num_parallel_calls=8 )

        if not is_predict:
            # shuffle before repeat and batch last
            dataset = dataset \
                .shuffle(MODEL_PARAMS['buffer_size'] ) \
                .repeat(MODEL_PARAMS['num_epochs'] ) \
                .batch( MODEL_PARAMS['batch_size'] )
        else:
            dataset = dataset \
                .batch( MODEL_PARAMS['batch_size'] )

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


def build_estimator_helper(model_fn, params):
    def build_estimator(model_dir, input_type):

        run_config = tf.estimator.RunConfig(
            save_summary_steps=50,
            log_step_count_steps=50,
            keep_checkpoint_max = 3,
            save_checkpoints_steps =50
        )

        if 'model_type' in params:
            # PNN -> PNN/IPNN
            model_dir = model_dir + '/' + params['model_type']

        if input_type not in model_fn:
            raise Exception('Only [{}] input_type are supported'.format(','.join(model_fn.keys()) ))

        estimator = tf.estimator.Estimator(
            model_fn = model_fn[input_type],
            config = run_config,
            params = params,
            model_dir= model_dir
        )

        return estimator
    return build_estimator


