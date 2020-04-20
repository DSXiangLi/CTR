from config import *
import tensorflow as tf


def build_features():
    f_dense = []
    Embedding_size = 4
    # categorical features
    for col, config in EMB_CONFIGS.items():
        ind = tf.feature_column.categorical_column_with_hash_bucket(col, hash_bucket_size = config['hash_size'])
        f_dense.append( tf.feature_column.embedding_column(ind, dimension = Embedding_size) )

    for col, config in BUCKET_CONFIGS.items():
        bucket = tf.feature_column.bucketized_column( tf.feature_column.numeric_column( col ), boundaries=config )
        f_dense.append( tf.feature_column.embedding_column(bucket, dimension = Embedding_size) )

    return f_dense
