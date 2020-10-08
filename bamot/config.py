import os
import sys
from pathlib import Path
from typing import Any, Dict, NamedTuple, Union

import yaml


class Config(NamedTuple):
    USING_CONFIG_FILE: bool
    CLUSTER_SIZE: Union[bool, float]
    KITTI_PATH: str
    DETECTIONS_PATH: str
    MAX_DIST: float
    MIN_LANDMARKS: int
    FEATURE_MATCHER: str
    NUM_FEATURES: str
    CONSTANT_MOTION: bool
    CONSTANT_MOTION_WEIGHT: float
    SUPERPOINT_WEIGHTS_PATH: str


MODULE_PATH = Path(os.path.dirname(__file__)).absolute()
SUPERPOINT_PATH: Path = MODULE_PATH / "thirdparty" / "SuperPoint"
SUPERPOINT_SETTINGS: Path = SUPERPOINT_PATH / "superpoint" / "settings.py"
# ./setup.sh script in thirdparty `SuperPoint` repo creates `settings.py`.
# The file won't be there if the script wasn't run and is not necessary
sys.path.append(SUPERPOINT_PATH.as_posix())
try:
    import superpoint.settings
except ModuleNotFoundError:
    with open(SUPERPOINT_SETTINGS, "w") as f:
        f.write('EXPER_PATH=""')

CONFIG_FILE = Path(".").parent / os.environ.get("CONFIG_FILE", default="config.yaml")
CONST_MOTION_WEIGHT_DEFAULT = float(os.environ.get("CONST_MOTION_WEIGHT", default=6))
CLUSTER_SIZE_DEFAULT = float(os.environ.get("CLUSTER_SIZE", default=8))

if CONFIG_FILE.exists():
    with open(CONFIG_FILE.as_posix(), "r") as fp:
        USER_CONFIG = yaml.load(fp, Loader=yaml.FullLoader)
else:
    USER_CONFIG: Dict[str, Any] = {}
kitti_path = Path(
    USER_CONFIG.get("kitti_path", "./data/KITTI/tracking/training")
).absolute()

CONFIG = Config(
    USING_CONFIG_FILE=CONFIG_FILE.exists(),
    CLUSTER_SIZE=USER_CONFIG.get("cluster_size", CLUSTER_SIZE_DEFAULT),
    KITTI_PATH=kitti_path.as_posix(),
    DETECTIONS_PATH=USER_CONFIG.get(
        "detections_path", (kitti_path / "preprocessed" / "mot").as_posix()
    ),
    CONSTANT_MOTION=USER_CONFIG.get("constant_motion", False),
    CONSTANT_MOTION_WEIGHT=USER_CONFIG.get(
        "constant_motion_weight", CONST_MOTION_WEIGHT_DEFAULT
    ),
    MAX_DIST=USER_CONFIG.get("max_dist", 150),
    MIN_LANDMARKS=USER_CONFIG.get("min_landmarks", 0),
    FEATURE_MATCHER=USER_CONFIG.get("feature_matcher", "orb"),
    NUM_FEATURES=USER_CONFIG.get("num_features", 8000),
    SUPERPOINT_WEIGHTS_PATH=USER_CONFIG.get(
        "superpoint_weights",
        (MODULE_PATH / "thirdparty" / "data" / "sp_v6").as_posix(),
    ),
)


def get_config_dict() -> Dict[str, Any]:
    return CONFIG._asdict()
