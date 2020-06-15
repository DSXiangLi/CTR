import pickle
import tensorflow as tf
import numpy as np
import os


class TFDump(object):
    def __init__(self, train_ratio):
        self.load_data() # data is already shuffled
        self.test_ratio = train_ratio
        self.nrecords = len( self.data )
        self.train, self.test = self.data[: int(self.nrecords * train_ratio)], self.data[int(self.nrecords * train_ratio): ]

    def load_data(self):
        with open( 'dataset.pkl', 'rb' ) as f:
            self.data = pickle.load( f )

    @staticmethod
    def int_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature( int64_list=tf.train.Int64List( value= value ) )

    @staticmethod
    def dump(data, type):
        with tf.python_io.TFRecordWriter('./data/amazon/amazon_{}.tfrecords'.format(type)) as writer:
            for record in data:
                example = tf.train.Example(
                    features = tf.train.Features(
                        feature = {
                        'reviewer_id': TFDump.int_feature(record[0]),
                        'hist_list': TFDump.int_feature(record[1]),
                        'item': TFDump.int_feature(record[2]),
                        'target': TFDump.int_feature(record[3])
                        }
                    )
                )

                writer.write(example.SerializeToString())

    def execute(self):
        TFDump.dump(self.train, 'train')
        TFDump.dump(self.test, 'test')



if __name__ == '__main__':
    preprocess = TFDump( train_ratio=0.8 )
    preprocess.execute()

