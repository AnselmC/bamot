import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from bamot import MODULE_PATH


@dataclass
class Config:
    USING_CONFIG_FILE: bool
    USING_MEDIAN_CLUSTER: bool
    USING_CONSTANT_MOTION: bool
    KITTI_PATH: str
    DETECTIONS_PATH: str
    MAX_DIST: float
    FEATURE_MATCHER: str
    NUM_FEATURES: str
    SLIDING_WINDOW_BA: int
    SLIDING_WINDOW_DESCRIPTORS: int
    MAX_BAD_FRAMES: int
    FRAME_RATE: int
    MAX_SPEED: float
    CONFIG_FILE: Optional[str] = None
    MIN_LANDMARKS: Optional[int] = None
    MAD_SCALE_FACTOR: Optional[float] = None
    CLUSTER_SIZE_CAR: Optional[float] = None
    CLUSTER_SIZE_PED: Optional[float] = None
    CONSTANT_MOTION_WEIGHTS_CAR: Optional[float] = None
    CONSTANT_MOTION_WEIGHTS_PED: Optional[float] = None
    SUPERPOINT_WEIGHTS_PATH: Optional[str] = None
    SUPERPOINT_PREPROCESSED_PATH: Optional[str] = None


# ENV VARIABLES
__config_file = Path(".").parent / os.environ.get("CONFIG_FILE", default="config.yaml")
__const_motion_weight_default = os.environ.get("CONST_MOTION_WEIGHT", default=1)
__const_motion_weights_default = [
    float(
        os.environ.get("CONST_MOTION_WEIGHT_ROT", default=__const_motion_weight_default)
    ),
    float(
        os.environ.get(
            "CONST_MOTION_WEIGHT_TRANS", default=__const_motion_weight_default
        )
    ),
]
__cluster_size_default = float(os.environ.get("CLUSTER_SIZE", default=8))
__mad_scale_factor_default = float(os.environ.get("MAD_SCALE_FACTOR", default=5.0))
__using_mad = bool(os.environ.get("USING_MAD", default=False))
__using_const_motion = bool(os.environ.get("USING_CONST_MOTION", default=True))
__kitti_scene = os.environ.get("SCENE", default="UNKNOWN")
__sliding_window_ba = int(os.environ.get("SLIDING_WINDOW_BA", default=10))

__default_max_speed = float(
    os.environ.get("MAX_SPEED_MS", default=40)
)  # about 140 km/h

__default_frame_rate = int(os.environ.get("FRAME_RATE", default=10))  # 10Hz for KITTI


if __config_file.exists():
    with open(__config_file.as_posix(), "r") as fp:
        __user_config = yaml.load(fp, Loader=yaml.FullLoader)
else:
    __user_config: Dict[str, Any] = {}

__kitti_path = Path(
    __user_config.get("kitti_path", "./data/KITTI/tracking/training")
).absolute()

__preprocessed_path = __kitti_path / "preprocessed"

CONFIG = Config(
    USING_CONFIG_FILE=__config_file.exists(),
    USING_MEDIAN_CLUSTER=__user_config.get("median_cluster", __using_mad),
    USING_CONSTANT_MOTION=__user_config.get("constant_motion", __using_const_motion),
    KITTI_PATH=__kitti_path.as_posix(),
    DETECTIONS_PATH=__user_config.get(
        "detections_path", (__kitti_path / "preprocessed" / "mot").as_posix()
    ),
    MAX_DIST=__user_config.get("max_dist", 150),
    FEATURE_MATCHER=__user_config.get("feature_matcher", "orb"),
    NUM_FEATURES=__user_config.get("num_features", 8000),
    SLIDING_WINDOW_BA=__user_config.get("sliding_window_ba", __sliding_window_ba),
    SLIDING_WINDOW_DESCRIPTORS=__user_config.get("sliding_window_desc", 10),
    MAX_BAD_FRAMES=__user_config.get("max_bad_frames", 3),
    MAX_SPEED=__user_config.get("max_speed_ms", __default_max_speed),
    FRAME_RATE=__user_config.get("frame_rate", __default_frame_rate),
)

if CONFIG.USING_CONFIG_FILE:
    CONFIG.CONFIG_FILE = __config_file.as_posix()

if CONFIG.USING_MEDIAN_CLUSTER:
    CONFIG.MAD_SCALE_FACTOR = __user_config.get(
        "mad_scale_factor", __mad_scale_factor_default
    )
else:
    _default_size = __user_config.get("cluster_size", __cluster_size_default)
    CONFIG.CLUSTER_SIZE_CAR = __user_config.get("cluster_size_car", _default_size)
    CONFIG.CLUSTER_SIZE_PED = __user_config.get("cluster_size_ped", _default_size)

if CONFIG.USING_CONSTANT_MOTION:
    _default_weight = __user_config.get(
        "const_motion_weights", __const_motion_weights_default
    )
    CONFIG.CONSTANT_MOTION_WEIGHTS_CAR = __user_config.get(
        "const_motion_weights_car", _default_weight
    )
    CONFIG.CONSTANT_MOTION_WEIGHTS_PED = __user_config.get(
        "const_motion_weights_ped", _default_weight
    )

if __user_config.get("min_landmarks"):
    CONFIG.MIN_LANDMARKS = __user_config.get("min_landmarks")

if CONFIG.FEATURE_MATCHER == "superpoint":
    CONFIG.SUPERPOINT_WEIGHTS_PATH = __user_config.get(
        "superpoint_weights",
        (MODULE_PATH / "thirdparty" / "data" / "sp_v6").as_posix(),
    )
elif CONFIG.FEATURE_MATCHER == "superpoint_preprocessed":
    CONFIG.SUPERPOINT_PREPROCESSED_PATH = __user_config.get(
        "superpoint_preprocessed_path",
        (__preprocessed_path / "superpoint" / __kitti_scene.zfill(4)).as_posix(),
    )


def get_config_dict() -> Dict[str, Any]:
    return CONFIG.__dict__
