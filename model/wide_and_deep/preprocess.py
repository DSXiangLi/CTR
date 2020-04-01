from config import *
import tensorflow as tf
from itertools import combinations

def build_features():
    f_onehot = []
    f_embedding = []
    f_numeric = []

    # categorical features
    for col, config in EMB_CONFIGS.items():
        ind = tf.feature_column.categorical_column_with_hash_bucket(col, hash_bucket_size = config['hash_size'])
        f_onehot.append( tf.feature_column.indicator_column(ind))
        f_embedding.append( tf.feature_column.embedding_column(ind, dimension = config['emb_size']) )

    # numeric features: both in numeric feature and bucketized to discrete feature
    for col, config in BUCKET_CONFIGS.items():
        num = tf.feature_column.numeric_column(col)
        f_numeric.append(num)
        bucket = tf.feature_column.bucketized_column( num, boundaries=config )
        f_onehot.append(bucket)

    # crossed features
    for col1,col2 in combinations(f_onehot,2):
        # if col is indicator of hashed bucuket, use raw feature directly
        if col1.parents[0].name in EMB_CONFIGS.keys():
            col1 = col1.parents[0].name
        if col2.parents[0].name in EMB_CONFIGS.keys():
            col2 = col2.parents[0].name

        crossed = tf.feature_column.crossed_column([col1, col2], hash_bucket_size = 20)
        f_onehot.append(tf.feature_column.indicator_column(crossed))

    # separate features into dense and sparse: sparse for LR, dense for MMLP
    f_dense = f_numeric + f_embedding
    f_sparse = f_onehot


    return f_sparse, f_dense

