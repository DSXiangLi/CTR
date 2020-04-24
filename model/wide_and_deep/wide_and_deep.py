"""
paper

Cheng H T, Koc L, Harmsen J, et al. Wide & deep learning for recommender systems[C]/
/Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. ACM, 2016: 7-10

"""
from model.wide_and_deep.preprocess import build_features
import tensorflow as tf


def build_estimator(model_dir, **kwargs):
    sparse_feature, dense_feature= build_features()

    run_config = tf.estimator.RunConfig(
        save_summary_steps=50,
        log_step_count_steps=50,
        keep_checkpoint_max = 3,
        save_checkpoints_steps =50
    )

    dnn_optimizer = tf.train.ProximalAdagradOptimizer(
                    learning_rate= 0.001,
                    l1_regularization_strength=0.001,
                    l2_regularization_strength=0.001
    )

    estimator = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=sparse_feature,
        dnn_feature_columns=dense_feature,
        dnn_optimizer = dnn_optimizer,
        dnn_dropout = 0.1,
        batch_norm = False,
        dnn_hidden_units = [48,32,16],
        config=run_config )

    return estimator