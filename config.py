# coding=utf-8
# project
TRAIN_DATA_PATH = 'data/image/train'
VALID_DATA_PATH = 'data/image/valid'
TEST_DATA_PATH = 'data/image/test'

RESULT_PATH = 'data/results'

DATA = {
        "CLASSES":[
                'alb',
                'bet',
                'dol',
                'lag',
                'other',
                'shark',
                'yft',
                ],
        "NUM":7
}

# DATA = {
#         "CLASSES":[
#                 'Astro',
#                 'Celcom',
#                 'Digi',
#                 'Hotlink',
#                 'IndahWater',
#                 'm1',
#                 'Maxis',
#                 'Others',
#                 'redONE',
#                 'SAJ',
#                 'Singtel',
#                 'SYABAS',
#                 'TIME',
#                 'TM',
#                 'TNB',
#                 'Umobile',
#                 'Unifi',
#                 'Vodafone',
#                 'yes',
#                 ],
#         "NUM":19
# }

# model
MODEL = {
        "ANCHORS":[[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj
            [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj
            [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]] ,# Anchors for big obj
        "STRIDES":[8, 16, 32],
        "ANCHORS_PER_SCLAE":3
}

# train
TRAIN = {
        "TRAIN_IMG_SIZE":416,
        "AUGMENT":True,
        "BATCH_SIZE":4,
        "MULTI_SCALE_TRAIN":True,
        "IOU_THRESHOLD_LOSS":0.5,
        "EPOCHS":50,
        "NUMBER_WORKERS":4,
        "MOMENTUM":0.9,
        "WEIGHT_DECAY":0.0005,
        "LR_INIT":1e-4,
        "LR_END":1e-6,
        "WARMUP_EPOCHS":2,  # or None
        "OPTIMIZER": 'sgd'     # or 'adam'
}


# test
TEST = {
        "TEST_IMG_SIZE":544,
        "BATCH_SIZE":1,         #1
        "NUMBER_WORKERS":0,     #0
        "CONF_THRESH":0.5,
        "NMS_THRESH":0.5,
        "MULTI_SCALE_TEST":False,
        "FLIP_TEST":False
}

# list of augments
AUGMENT = {
        "DILATION": True,
        "SMUDGE": False,
        "RANDOM_CHANGE_COLOR": False,
        "RANDOM_HORIZONTAL_FLIP": False,
        "RANDOM_CROP": True,
        "RANDOM_AFFINE": True,
        "RANDOM_ROTATE": False,
        "RANDOM_ROTATE_RANGE": 3,                     # degree rotation
}