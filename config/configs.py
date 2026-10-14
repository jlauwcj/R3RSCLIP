import yaml
from easydict import EasyDict as edict

cfg = edict()

cfg.DATASET = edict()
cfg.DATASET.NAME = ''
cfg.DATASET.NUM_CLASSES = 0
cfg.DATASET.REDUCE_ZERO_LABEL = True
cfg.DATASET.DATAROOT = ''
cfg.DATASET.SCALE = []
cfg.DATASET.RATIO_RANGE = []
cfg.DATASET.CROP_SIZE = []
cfg.DATASET.CAT_MAX_RATIO = 0
cfg.DATASET.TEXT_WEIGHT = ''
cfg.DATASET.IMG_NORM_CFG = edict()
cfg.DATASET.IMG_NORM_CFG.MEAN = []
cfg.DATASET.IMG_NORM_CFG.STD = []
cfg.DATASET.IMG_NORM_CFG.RGB = True
cfg.DATASET.K = 0
cfg.DATASET.DISTILL_K = 0
cfg.DATASET.THRESHOLD = 0
cfg.DATASET.IGNORE_INDEX = 255
cfg.DATASET.PALETTE = []
cfg.DATASET.EVAL_EXCLUDE_INDEXES = []

cfg.MODEL = edict()
cfg.MODEL.USE_MULTISCALE = False
cfg.MODEL.MULTISCALE_COMPRESS_DIM = 256
cfg.MODEL.MULTISCALE_WARMUP_EPOCHS = 10
cfg.MODEL.MULTISCALE_LR_MULTIPLIER = 1.0
cfg.MODEL.MAX_SCALE = 0.5
cfg.MODEL.USE_CONSISTENCY = False
cfg.MODEL.CONS_WEIGHT = 1.0            #  (0.5 - 1.0)
cfg.MODEL.CONS_START_EPOCH = 5
cfg.MODEL.USE_FASB = False
cfg.MODEL.FASB_LOW_FREQ_THRESHOLD = 0.2
cfg.MODEL.FASB_LR_MULTIPLIER = 1.0 
cfg.MODEL.FEATURE_EXTRACTOR = ''
cfg.MODEL.TEXT_CHANNEL = 0
cfg.MODEL.VISUAL_CHANNEL = 0
cfg.MODEL.TRAINING = False

cfg.TRAIN = edict()
cfg.TRAIN.BATCH_SIZE = 1
cfg.TRAIN.MAX_EPOCH = 50
cfg.TRAIN.EPOCH = 0
cfg.TRAIN.MAX_ITER = 0
cfg.TRAIN.LR = 0
cfg.TRAIN.LOG = ''

cfg.TEST = edict()
cfg.TEST.BATCH_SIZE = 0
cfg.TEST.PD = 0
cfg.TEST.ReCLIP_PD = 0.5

cfg.EVAL_METRIC = ''
cfg.SAVE_DIR = ''
cfg.PSEUDO_DIR = ''
cfg.NUM_WORKERS = 0
cfg.LOAD_PATH = ''
cfg.LOAD_DISTILL_PATH = ''

def merge_a_to_b(a, b):
    if type(a) is not edict:
        return
    for k in a:
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))
        if type(a[k]) is edict:
            merge_a_to_b(a[k], b[k])
        else:
            b[k] = a[k]
    return cfg


def cfg_from_file(filename):

    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    merge_a_to_b(yaml_cfg, cfg)
    return cfg
