CENSUS_FEATURE_NAME =[
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

CENSUS_TARGET = 'income_bracket'

CENSUS_TARGET_VAL = '>50K'

CENSUS_CSV_RECORD_DEFAULTS = [[0.0], [''], [0.0], [''], [0.0], [''], [''], [''], [''], [''],
                        [0.0], [0.0], [0.0], [''], ['']]

CENSUS_EMB_CONFIGS = {
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

CENSUS_BUCKET_CONFIGS = {
    'age':{
        'bin':[18, 25, 30, 35, 40, 45, 50, 55, 60, 65],
        'emb_size':4
    },
    'fnlwgt':{
        'bin':[6*(10**4), 10**5, 1.3*(10**5), 1.5*(10**5), 1.7*(10**5), 1.9*(10**5),
              2.1*(10**5), 2.5*(10**5), 3*(10**5)],
        'emb_size':4
    },
    'education_num' : {
        'bin':[7,8,10,11,13],
        'emb_size':4
    },
    'hours_per_week':{
        'bin':[25,35,40,45,55],
        'emb_size':4
    },
    'capital_gain':{
        'bin':[0,1],
        'emb_size':4
    },
    'capital_loss':{
        'bin':[0,1],
        'emb_size':4
    }
}

CENSUS_NORM_CONFIGS = {
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



