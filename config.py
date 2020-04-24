import tensorflow as tf

def data_dir(name, input_type):
    DATA_CSV_DIR = './data/adult/{}.csv'
    DATA_LIBSVM_DIR = './data/criteo/{}.libsvm'
    if input_type == 'dense':
        return DATA_CSV_DIR.format(name)
    elif input_type == 'sparse':
        return DATA_LIBSVM_DIR.format( name )
    else:
        raise Exception('Currenlty only dense or sparse is supported')

MODEL_DIR = './test_checkpoint/{}'

FEATURE_NAME =[
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

TARGET = 'income_bracket'

CSV_RECORD_DEFAULTS = [[0.0], [''], [0.0], [''], [0.0], [''], [''], [''], [''], [''],
                        [0.0], [0.0], [0.0], [''], ['']]
DTYPE ={}
for i ,j in enumerate(CSV_RECORD_DEFAULTS):
    if j[0]=='':
        DTYPE[FEATURE_NAME[i]] = tf.string
    else :
        DTYPE[FEATURE_NAME[i]] = tf.float32


MODEL_PARAMS = {
    'batch_size':512,
    'num_epochs':5000,
    'buffer_size':512
}

EMB_CONFIGS = {
    'workclass':{
        'hash_size':10,
        'emb_size':4
    },
    'education':{
        'hash_size':10,
        'emb_size':4
    },
    'marital_status':{
        'hash_size':10,
        'emb_size':4
    },
    'occupation': {
        'hash_size': 30,
        'emb_size': 4
    },
    'relationship': {
        'hash_size': 10,
        'emb_size': 4
    },
    'race': {
        'hash_size': 10,
        'emb_size': 4
    },
    'gender': {
        'hash_size': 10,
        'emb_size': 4
    },
    'native_country':{
        'hash_size':30,
        'emb_size': 4
    }
}

BUCKET_CONFIGS = {
    'age':[18, 25, 30, 35, 40, 45, 50, 55, 60, 65],
    'fnlwgt':[6*(10**4), 10**5, 1.3*(10**5), 1.5*(10**5), 1.7*(10**5), 1.9*(10**5),
              2.1*(10**5), 2.5*(10**5), 3*(10**5)],
    'education_num' : [7,8,10,11,13],
    'hours_per_week':[25,35,40,45,55],
    'capital_gain':[0,1],
    'capital_loss':[0,1]
}

NORM_CONFIGS = {
    'age':{
        'mean': 38.5,
        'std': 13.6
    },
    'fnlwgt':{
        'mean':189781.81,
        'std':105549.76
    },
    'education_num':{
        'mean':10.08,
        'std':2.57
    },
    'hours_per_week':{
        'mean':40.44,
        'std':12.35
    },
    'capital_gain':{
        'mean':1077.62,
        'std':7385.40
    },
    'capital_loss':{
        'mean':87.31,
        'std':402.97
    }
}
