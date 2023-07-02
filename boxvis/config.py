from detectron2.config import CfgNode as CN


def add_boxvis_config(cfg):
    cfg.DATASETS.DATASET_RATIO = []

    # DataLoader
    cfg.INPUT.SAMPLING_FRAME_NUM = 2
    cfg.INPUT.SAMPLING_FRAME_RANGE = 20
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = []  # "brightness", "contrast", "saturation", "rotation"

    cfg.INPUT.MIN_SIZE_TRAIN = (352, 384, 416, 448, 512, 544, 576, 608, 640)
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.CROP = CN()
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.TYPE = "absolute_range"
    cfg.INPUT.CROP.SIZE = (384, 600)

    # Pseudo Data Use
    cfg.INPUT.PSEUDO = CN()
    cfg.INPUT.PSEUDO.AUGMENTATIONS = ['rotation']
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN = (352, 384, 416, 448, 512, 544, 576, 608, 640)  # , 672, 704, 736, 768)
    cfg.INPUT.PSEUDO.MAX_SIZE_TRAIN = 768
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN_SAMPLING = "choice_by_clip"
    cfg.INPUT.PSEUDO.CROP = CN()
    cfg.INPUT.PSEUDO.CROP.ENABLED = False
    cfg.INPUT.PSEUDO.CROP.TYPE = "absolute_range"
    cfg.INPUT.PSEUDO.CROP.SIZE = (384, 600)

    # LSJ
    cfg.INPUT.LSJ_AUG = CN()
    cfg.INPUT.LSJ_AUG.ENABLED = False
    cfg.INPUT.LSJ_AUG.IMAGE_SIZE = 1024
    cfg.INPUT.LSJ_AUG.MIN_SCALE = 0.1
    cfg.INPUT.LSJ_AUG.MAX_SCALE = 2.0

    # VIT transformer backbone
    cfg.MODEL.VIT = CN()
    cfg.MODEL.VIT.PRETRAIN_IMG_SIZE = 1024
    cfg.MODEL.VIT.PATCH_SIZE = 16
    cfg.MODEL.VIT.EMBED_DIM = 768
    cfg.MODEL.VIT.DEPTH = 12
    cfg.MODEL.VIT.NUM_HEADS = 12
    cfg.MODEL.VIT.MLP_RATIO = 4.0
    cfg.MODEL.VIT.OUT_CHANNELS = 256
    cfg.MODEL.VIT.QKV_BIAS = True
    cfg.MODEL.VIT.USE_ABS_POS = True
    cfg.MODEL.VIT.USE_REL_POS = False
    cfg.MODEL.VIT.REL_POS_ZERO_INIT = False
    cfg.MODEL.VIT.WINDOW_SIZE = 0
    cfg.MODEL.VIT.GLOBAL_ATTN_INDEXES = ()
    cfg.MODEL.VIT.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.VIT.USE_CHECKPOINT = False

    # BoxVIS
    cfg.MODEL.BoxVIS = CN()
    cfg.MODEL.BoxVIS.BoxVIS_ENABLED = True
    cfg.MODEL.BoxVIS.BOTTOM_PIXELS_REMOVED = 10
    cfg.MODEL.BoxVIS.PAIRWISE_ENABLED = True
    cfg.MODEL.BoxVIS.PAIRWISE_STPAIR_NUM = 7  # 2 sptial neighbours and 5 temporal neighbours
    cfg.MODEL.BoxVIS.PAIRWISE_SIZE = 3
    cfg.MODEL.BoxVIS.PAIRWISE_DILATION = 4
    cfg.MODEL.BoxVIS.PAIRWISE_COLOR_THRESH = 0.3  # 0.3 => (dR+dG+dB) < 2, 0.15 => (dR+dG+dB) < 4
    cfg.MODEL.BoxVIS.PAIRWISE_PATCH_KERNEL_SIZE = 3
    cfg.MODEL.BoxVIS.PAIRWISE_PATCH_STRIDE = 2
    cfg.MODEL.BoxVIS.PAIRWISE_PATCH_THRESH = 0.9

    # Teacher Net
    cfg.MODEL.BoxVIS.EMA_ENABLED = True
    cfg.MODEL.BoxVIS.PSEUDO_MASK_SCORE_THRESH = 0.5

    # BVISD dataset
    cfg.MODEL.BoxVIS.BVISD_ENABLED = True

    # Inference
    cfg.MODEL.BoxVIS.TEST = CN()
    cfg.MODEL.BoxVIS.TEST.TRACKER_TYPE = 'minvis'  # 'minvis' => frame-level tracker, 'mdqe' => clip-level tracker
    cfg.MODEL.BoxVIS.TEST.WINDOW_INFERENCE = False
    cfg.MODEL.BoxVIS.TEST.MULTI_CLS_ON = True
    cfg.MODEL.BoxVIS.TEST.APPLY_CLS_THRES = 0.05
    cfg.MODEL.BoxVIS.TEST.MERGE_ON_CPU = False

    # clip-by-clip tracking with overlapped frames
    cfg.MODEL.BoxVIS.TEST.NUM_FRAMES = 3
    cfg.MODEL.BoxVIS.TEST.NUM_FRAMES_WINDOW = 30
    cfg.MODEL.BoxVIS.TEST.NUM_MAX_INST = 50
    cfg.MODEL.BoxVIS.TEST.CLIP_STRIDE = 1




