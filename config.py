import tensorflow as tf

DATA_DIR = './data/adult/{}.csv'
MODEL_DIR = './checkpoint/{}/'

FEATURE_NAME =[
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

TARGET = 'income_bracket'

CSV_RECORD_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]
DTYPE ={}
for i ,j in enumerate(CSV_RECORD_DEFAULTS):
    if j[0]=='':
        DTYPE[FEATURE_NAME[i]] = tf.string
    else :
        DTYPE[FEATURE_NAME[i]] = tf.float32


MODEL_PARAMS = {
    'batch_size':512,
    'num_epochs':500,
    'buffer_size':512
}

EMB_CONFIGS = {
    'workclass':{
        'hash_size':10,
        'emb_size':5
    },
    'education':{
        'hash_size':10,
        'emb_size':5
    },
    'marital_status':{
        'hash_size':10,
        'emb_size':5
    },
    'occupation': {
        'hash_size': 100,
        'emb_size': 5
    },
    'relationship': {
        'hash_size': 10,
        'emb_size': 5
    },
    'race': {
        'hash_size': 10,
        'emb_size': 5
    },
    'gender': {
        'hash_size': 10,
        'emb_size': 2
    },
    'native_country':{
        'hash_size':100,
        'emb_size': 10
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


