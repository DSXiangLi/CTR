from model.wide_and_deep.preprocess import build_features
import tensorflow as tf


def build_estimator(model_dir):
    sparse_feature, dense_feature= build_features()

    run_config = tf.estimator.RunConfig(
        save_summary_steps=100,
        log_step_count_steps=100,
        keep_checkpoint_max = 3,
        save_checkpoints_steps =100
    )

    estimator = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=sparse_feature,
        dnn_feature_columns=dense_feature,
        dnn_hidden_units = [64,32,16],
        config=run_config )

    return estimator