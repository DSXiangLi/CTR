from config import *
import tensorflow as tf

def build_features():
    f_one_hot = []
    for col, config in EMB_CONFIGS.items():
        ind = tf.feature_column.categorical_column_with_hash_bucket(col, hash_bucket_size = config['hash_size'])
        one_hot = tf.feature_column.indicator_column(ind)
        f_one_hot.append(one_hot)

    for col, config in BUCKET_CONFIGS.items():
        num = tf.feature_column.numeric_column(col)
        bucket = tf.feature_column.bucketized_column(num, boundaries = config)
        one_hot = tf.feature_column.indicator_column(bucket)
        f_one_hot.append(one_hot)

    return f_one_hot