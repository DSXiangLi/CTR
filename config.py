"""
Configuration for each dataset

"""

import os

class CONFIG:
    """
        des:
        data_dir
        checkpoint_dir: data/model
        input_parser: libsvm/csv/tfrecord
        padded_shape: if varlen feature included, pad shape is needed for padded_batch
    """
    CHECKPOINT_DIR = './{}_checkpoint/{}'

    DATA_MAP = {
        'census': '{}.csv',
        'frappe': 'frappe.{}.libfm',
        'amazon': 'amazon_{}.tfrecords'
    }

    PARSER_MAP = {
        'census': 'csv',
        'frappe': 'libsvm',
        'amazon': 'tfrecord'
    }

    TYPE_MAP = {
        'census': 'dense',
        'frappe': 'sparse',
        'amazon': 'varlen-sparse'
    }

    PADDED_SHAPE = {
        'census': None,
        'frappe': None,
        'amazon': ({
                'reviewer_id': [],
                'hist_item_list': [None],
                'hist_category_list':[None],
                'hist_length': [],
                'item': [],
                'item_category':[],
        },[1])
    }

    def __init__(self, model_name, data_name):
        self.model_name = model_name
        self.data_name = data_name
        self.input_check()

    def input_check(self):
        if self.data_name not in CONFIG.DATA_MAP.keys():
            raise Exception( 'Currenlty only [{}] is supported'.format( ' | '.join( CONFIG.DATA_MAP.keys() ) ) )

    @property
    def data_dir(self):
        return os.path.join('data', self.data_name, CONFIG.DATA_MAP[self.data_name])

    @property
    def checkpoint_dir(self):
        return CONFIG.CHECKPOINT_DIR.format(self.data_name, self.model_name)

    @property
    def input_parser(self):
        return CONFIG.PARSER_MAP[self.data_name]

    @property
    def pad_shape(self):
        return CONFIG.PADDED_SHAPE[self.data_name]

    @property
    def input_type(self):
        return CONFIG.TYPE_MAP[self.data_name]

    def get_constZ(self):
        # get const for dataset: defined in const/dataset
        pass

MODEL_PARAMS = {
    'batch_size': 512,
    'num_epochs': 5000,
    'buffer_size': 512
}