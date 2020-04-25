from config import *
import tensorflow as tf

def build_features():
    f_one_hot = []
    field_dict = []
    field = 0
    features = 0

    for col, config in BUCKET_CONFIGS.items():
        num = tf.feature_column.numeric_column(col)
        bucket = tf.feature_column.bucketized_column(num, boundaries = config['bin'])
        one_hot = tf.feature_column.indicator_column(bucket)
        f_one_hot.append(one_hot)

        field_dict += [ (i,field)for i in range(features, features+len(config['bin'])+1)]
        features += (len(config['bin'])+1)
        field +=1

    field_dict = dict(field_dict)

    return f_one_hot, field_dict