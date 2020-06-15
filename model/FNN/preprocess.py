from const import *
import tensorflow as tf

def build_features():
    f_sparse = []

    for col, config in EMB_CONFIGS.items():
        ind = tf.feature_column.categorical_column_with_hash_bucket(col, hash_bucket_size = config['hash_size'])
        one_hot = tf.feature_column.indicator_column(ind)
        f_sparse.append(one_hot)

    for col, config in BUCKET_CONFIGS.items():
        num = tf.feature_column.numeric_column( col )
        bucket = tf.feature_column.bucketized_column( num, boundaries = config['bin'] )
        f_sparse.append( bucket )

    return f_sparse

