import tensorflow as tf
import pickle
AMAZON_PARAMS = {
    'field_size':10 ,
    'feature_size':10,
    'embedding_size':10
}

AMAZON_PROTO = {
    'reviewer_id': tf.FixedLenFeature( [], tf.int64 ),
    'hist_list': tf.VarLenFeature( tf.int64 ),
    'item': tf.FixedLenFeature( [], tf.int64 ),
    'target': tf.FixedLenFeature( [], tf.int64 )
}


AMAZON_VARLEN = ['hist_list']


with open('data/amazon/remap.pkl', 'rb') as f:
    _ = pickle.load(f) # uid, iid
    AMAZON_CATE_LIST  = pickle.load(f)
