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
    MIN_LANDMARKS: Optional[int] = None
    MAD_SCALE_FACTOR: Optional[float] = None
    CLUSTER_SIZE_CAR: Optional[float] = None
    CLUSTER_SIZE_PED: Optional[float] = None
    CONSTANT_MOTION_WEIGHT_CAR: Optional[float] = None
    CONSTANT_MOTION_WEIGHT_PED: Optional[float] = None
    SUPERPOINT_WEIGHTS_PATH: Optional[str] = None
    SUPERPOINT_PREPROCESSED_PATH: Optional[str] = None


# ENV VARIABLES
__config_file = Path(".").parent / os.environ.get("CONFIG_FILE", default="config.yaml")
__const_motion_weight_default = float(os.environ.get("CONST_MOTION_WEIGHT", default=6))
__cluster_size_default = float(os.environ.get("CLUSTER_SIZE", default=8))
__mad_scale_factor_default = float(os.environ.get("MAD_SCALE_FACTOR", default=1.4))
__kitti_scene = os.environ.get("SCENE", default="UNKNOWN")

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
    USING_MEDIAN_CLUSTER=__user_config.get("median_cluster", False),
    USING_CONSTANT_MOTION=__user_config.get("constant_motion", False),
    KITTI_PATH=__kitti_path.as_posix(),
    DETECTIONS_PATH=__user_config.get(
        "detections_path", (__kitti_path / "preprocessed" / "mot").as_posix()
    ),
    MAX_DIST=__user_config.get("max_dist", 150),
    FEATURE_MATCHER=__user_config.get("feature_matcher", "orb"),
    NUM_FEATURES=__user_config.get("num_features", 8000),
)

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
        "const_motion_weight", __const_motion_weight_default
    )
    CONFIG.CONSTANT_MOTION_WEIGHT_CAR = __user_config.get(
        "const_motion_car", _default_weight
    )
    CONFIG.CONSTANT_MOTION_WEIGHT_PED = __user_config.get(
        "const_motion_ped", _default_weight
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
