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


def _build_config(config_file=Path(".").parent / "config.yaml") -> Config:

    if config_file.exists():
        with open(config_file.as_posix(), "r") as fp:
            user_config = yaml.load(fp, Loader=yaml.FullLoader)
    else:
        user_config: Dict[str, Any] = {}
    kitti_path = Path(
        user_config.get("kitti_path", "./data/KITTI/tracking/training")
    ).absolute()
    return Config(
        USING_CONFIG_FILE=config_file.exists(),
        CLUSTER_SIZE=user_config.get("cluster_size", 6.0),
        KITTI_PATH=kitti_path.as_posix(),
        DETECTIONS_PATH=user_config.get(
            "detections_path", (kitti_path / "preprocessed" / "mot").as_posix()
        ),
        CONSTANT_MOTION=user_config.get("constant_motion", False),
        CONSTANT_MOTION_WEIGHT=user_config.get("constant_motion_weight", 6.0),
        MAX_DIST=user_config.get("max_dist", 150),
        FEATURE_MATCHER=user_config.get("feature_matcher", "orb"),
        NUM_FEATURES=user_config.get("num_features", 8000),
        SUPERPOINT_WEIGHTS_PATH=user_config.get(
            "superpoint_weights",
            (MODULE_PATH / "thirdparty" / "data" / "sp_v6").as_posix(),
        ),
    )


CONFIG = _build_config()


def update_config(config_file: Path):
    global CONFIG
    CONFIG = _build_config(config_file)


def get_config_dict() -> Dict[str, Any]:
    return CONFIG._asdict()
