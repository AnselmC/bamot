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
CONFIG_FILE = Path(".").parent / os.environ.get("CONFIG_FILE", default="config.yaml")
CONST_MOTION_WEIGHT_DEFAULT = float(os.environ.get("CONST_MOTION_WEIGHT", default=6))
CLUSTER_SIZE_DEFAULT = float(os.environ.get("CLUSTER_SIZE", default=8))
USE_MEDIAN_CLUSTER = bool(os.environ.get("USE_MEDIAN_CLUSTER", default=False))
MAD_SCALE_FACTOR_DEFAULT = float(os.environ.get("MAD_SCALE_FACTOR", default=1.4))
KITTI_SCENE = os.environ.get("SCENE", default=None)

if CONFIG_FILE.exists():
    with open(CONFIG_FILE.as_posix(), "r") as fp:
        USER_CONFIG = yaml.load(fp, Loader=yaml.FullLoader)
else:
    USER_CONFIG: Dict[str, Any] = {}

_kitti_path = Path(
    USER_CONFIG.get("kitti_path", "./data/KITTI/tracking/training")
).absolute()

_preprocessed_path = _kitti_path / "preprocessed"

CONFIG = Config(
    USING_CONFIG_FILE=CONFIG_FILE.exists(),
    USING_MEDIAN_CLUSTER=USER_CONFIG.get("median_cluster", USE_MEDIAN_CLUSTER),
    USING_CONSTANT_MOTION=USER_CONFIG.get("constant_motion", False),
    KITTI_PATH=_kitti_path.as_posix(),
    DETECTIONS_PATH=USER_CONFIG.get(
        "detections_path", (_kitti_path / "preprocessed" / "mot").as_posix()
    ),
    MAX_DIST=USER_CONFIG.get("max_dist", 150),
    FEATURE_MATCHER=USER_CONFIG.get("feature_matcher", "orb"),
    NUM_FEATURES=USER_CONFIG.get("num_features", 8000),
)

if CONFIG.USING_MEDIAN_CLUSTER:
    CONFIG.MAD_SCALE_FACTOR = USER_CONFIG.get(
        "mad_scale_factor", MAD_SCALE_FACTOR_DEFAULT
    )
else:
    _default_size = USER_CONFIG.get("cluster_size", CLUSTER_SIZE_DEFAULT)
    CONFIG.CLUSTER_SIZE_CAR = USER_CONFIG.get("cluster_size_car", _default_size)
    CONFIG.CLUSTER_SIZE_PED = USER_CONFIG.get("cluster_size_ped", _default_size)

if CONFIG.USING_CONSTANT_MOTION:
    _default_weight = USER_CONFIG.get(
        "const_motion_weight", CONST_MOTION_WEIGHT_DEFAULT
    )
    CONFIG.CONSTANT_MOTION_WEIGHT_CAR = USER_CONFIG.get(
        "const_motion_car", _default_weight
    )
    CONFIG.CONSTANT_MOTION_WEIGHT_PED = USER_CONFIG.get(
        "const_motion_ped", _default_weight
    )

if USER_CONFIG.get("min_landmarks"):
    CONFIG.MIN_LANDMARKS = USER_CONFIG.get("min_landmarks")

if CONFIG.FEATURE_MATCHER == "superpoint":
    CONFIG.SUPERPOINT_WEIGHTS_PATH = USER_CONFIG.get(
        "superpoint_weights",
        (MODULE_PATH / "thirdparty" / "data" / "sp_v6").as_posix(),
    )
    CONFIG.SUPERPOINT_PREPROCESSED_PATH = USER_CONFIG.get(
        "superpoint_preprocessed_path",
        (_preprocessed_path / "superpoint" / KITTI_SCENE.zfill(4)).as_posix(),
    )


def get_config_dict() -> Dict[str, Any]:
    return CONFIG.__dict__
