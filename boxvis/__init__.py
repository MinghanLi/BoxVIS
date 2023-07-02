# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.

# config
from .config import add_boxvis_config

from .utils.copy_TeacherNet_weights import copy_TeacherNet_weights

from .data import *

from .boxvis_video_mask2former_model import BoxVIS_VideoMaskFormer

