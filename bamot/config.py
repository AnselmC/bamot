import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from bamot import MODULE_PATH


class ConfigError(Exception):
    pass


@dataclass
class Config:
    USING_CONFIG_FILE: bool
    USING_MEDIAN_CLUSTER: bool
    USING_CONSTANT_MOTION: bool
    KITTI_PATH: str
    GT_DETECTIONS_PATH: str
    EST_DETECTIONS_PATH: str
    MAX_DIST: float
    FEATURE_MATCHER: str
    NUM_FEATURES: str
    SLIDING_WINDOW_BA: int
    SLIDING_WINDOW_DESCRIPTORS: int
    SLIDING_WINDOW_DIR_VEC: int
    FRAME_RATE: int
    MAX_SPEED_CAR: float
    MAX_SPEED_PED: float
    MIN_LANDMARKS_CAR: int
    MIN_LANDMARKS_PED: int
    BA_EVERY_N_STEPS: int
    BA_NORMALIZE_TRANS_ERROR: bool
    KEEP_TRACK_FOR_N_FRAMES_AFTER_LOST: int
    TRUST_2D: str
    SAVE_UPDATED_2D_TRACK: bool
    SAVE_3D_TRACK: bool
    SAVE_OBB_DATA: bool
    CAR_DIMS: Tuple[float, float, float]  # HxWxL
    PED_DIMS: Tuple[float, float, float]
    FORCE_NEW_DETECTIONS: bool
    FINAL_FULL_BA: bool = False
    MAX_MAX_DIST_MULTIPLIER: int = 5
    TRACK_POINT_CLOUD_SIZES: bool = False
    CONFIG_FILE: Optional[str] = None
    MAD_SCALE_FACTOR: Optional[float] = None
    CLUSTER_RADIUS_CAR: Optional[float] = None
    CLUSTER_RADIUS_PED: Optional[float] = None
    CONSTANT_MOTION_WEIGHTS_CAR: Optional[float] = None
    CONSTANT_MOTION_WEIGHTS_PED: Optional[float] = None
    SUPERPOINT_WEIGHTS_PATH: Optional[str] = None
    SUPERPOINT_PREPROCESSED_PATH: Optional[str] = None


# ENV VARIABLES
__config_file_default = Path(".").parent / os.environ.get(
    "CONFIG_FILE", default="config.yaml"
)
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
__track_point_cloud_sizes_default = bool(
    int(os.environ.get("TRACK_POINT_CLOUD_SIZES", default=0))
)
__save_updated_2d_track_default = bool(
    int(os.environ.get("SAVE_UPDATED_2D_TRACK", default=0))
)
__save_3d_track_default = bool(int(os.environ.get("SAVE_3D_TRACK", default=1)))
__save_obb_data_default = bool(int(os.environ.get("SAVE_OBB_DATA", default=0)))
__final_full_ba_default = bool(int(os.environ.get("FINAL_FULL_BA", default=0)))
__cluster_radius_default = float(os.environ.get("CLUSTER_RADIUS", default=8))
__mad_scale_factor_default = float(os.environ.get("MAD_SCALE_FACTOR", default=5.0))
__using_mad_default = bool(int(os.environ.get("USING_MAD", default=0)))
__using_const_motion_default = bool(
    int(os.environ.get("USING_CONST_MOTION", default=1))
)
__kitti_scene_default = os.environ.get("SCENE", default="UNKNOWN")
__sliding_window_ba_default = int(os.environ.get("SLIDING_WINDOW_BA", default=10))
__sliding_window_dir_default = int(os.environ.get("SLIDING_WINDOW_DIR_VEC", default=10))
__ba_every_n_steps_default = int(os.environ.get("BA_EVERY_N_STEPS", default=1))
__ba_normalize_trans_error = bool(
    int(os.environ.get("BA_NORMALIZE_TRANS_ERROR", default=0))
)
__keep_track_for_n_frames_after_lost_default = int(
    os.environ.get("KEEP_TRACK_FOR_N_FRAMES_AFTER_LOST", default=1)
)
__min_landmarks_default = int(os.environ.get("MIN_LANDMARKS", default=1))

__max_speed_car_default = float(
    os.environ.get("MAX_SPEED_CAR", default=60)
)  # about 140 km/h
__max_speed_ped_default = float(
    os.environ.get("MAX_SPEED_PED", default=8)
)  # about 14 m/h

__frame_rate_default = int(os.environ.get("FRAME_RATE", default=5))  # 10Hz for KITTI

__trust_2d_default = str(os.environ.get("TRUST_2D", default="yes"))

__car_dims_default = (1.6, 1.8, 4.3)  # HxWxL
__ped_dims_default = (1.9, 0.7, 0.9)

__force_new_detections_default = bool(
    int(os.environ.get("FORCE_NEW_DETECTIONS", default=0))
)

__kitti_tracking_path_default = Path(
    os.environ.get(
        "KITTI_TRACKING_PATH",
        (Path(__file__).parent.parent / "data" / "KITTI" / "tracking"),
    )
)

__kitti_tracking_subset_default = os.environ.get(
    "KITTI_TRACKING_SUBSET", default="training"
)


# READ CONFIG FILE
if __config_file_default.exists():
    with open(__config_file_default.as_posix(), "r") as fp:
        __user_config = yaml.load(fp, Loader=yaml.FullLoader)
else:
    __user_config = {}


__kitti_tracking_path = Path(
    __user_config.get("kitti_tracking_path", __kitti_tracking_path_default)
)

__kitti_tracking_subset = __user_config.get(
    "kitti_tracking_subset", __kitti_tracking_subset_default
)

__kitti_path = __kitti_tracking_path / __kitti_tracking_subset

__preprocessed_path_default = __kitti_path / "stereo_detections"

CONFIG = Config(
    USING_CONFIG_FILE=__config_file_default.exists(),
    USING_MEDIAN_CLUSTER=__user_config.get("median_cluster", __using_mad_default),
    USING_CONSTANT_MOTION=__user_config.get(
        "constant_motion", __using_const_motion_default
    ),
    KITTI_PATH=__kitti_path.as_posix(),  # needs to be JSON serializable s.t. config can be dumped
    GT_DETECTIONS_PATH=__user_config.get(
        "gt_detections_path", (__kitti_path / "stereo_detections_gt").as_posix(),
    ),
    EST_DETECTIONS_PATH=__user_config.get(
        "detections_path", (__kitti_path / "stereo_detections").as_posix(),
    ),
    MAX_DIST=__user_config.get("max_dist", 65),
    FEATURE_MATCHER=__user_config.get("feature_matcher", "orb"),
    NUM_FEATURES=__user_config.get("num_features", 8000),
    BA_EVERY_N_STEPS=__user_config.get("ba_every_n_steps", __ba_every_n_steps_default),
    BA_NORMALIZE_TRANS_ERROR=__user_config.get(
        "ba_normalize_trans_error", __ba_normalize_trans_error
    ),
    SLIDING_WINDOW_BA=__user_config.get(
        "sliding_window_ba", __sliding_window_ba_default
    ),
    SLIDING_WINDOW_DIR_VEC=__user_config.get(
        "sliding_window_dir_vec", __sliding_window_dir_default
    ),
    TRUST_2D=__user_config.get("trust_2d", __trust_2d_default),
    SLIDING_WINDOW_DESCRIPTORS=__user_config.get("sliding_window_desc", 10),
    MAX_SPEED_CAR=__user_config.get("max_speed_car", __max_speed_car_default),
    MAX_SPEED_PED=__user_config.get("max_speed_ped", __max_speed_ped_default),
    MIN_LANDMARKS_PED=__user_config.get(
        "min_landmarks_ped", __user_config.get("min_landmarks", __min_landmarks_default)
    ),
    MIN_LANDMARKS_CAR=__user_config.get(
        "min_landmarks_car", __user_config.get("min_landmarks", __min_landmarks_default)
    ),
    FRAME_RATE=__user_config.get("frame_rate", __frame_rate_default),
    TRACK_POINT_CLOUD_SIZES=__user_config.get(
        "track_point_cloud_sizes", __track_point_cloud_sizes_default
    ),
    SAVE_UPDATED_2D_TRACK=__user_config.get(
        "save_updated_2d_track", __save_updated_2d_track_default
    ),
    SAVE_3D_TRACK=__user_config.get("save_3d_track", __save_3d_track_default),
    SAVE_OBB_DATA=__user_config.get("save_obb_data", __save_obb_data_default),
    FINAL_FULL_BA=__user_config.get("final_full_ba", __final_full_ba_default),
    KEEP_TRACK_FOR_N_FRAMES_AFTER_LOST=__user_config.get(
        "keep_track_for_n_frames_after_lost",
        __keep_track_for_n_frames_after_lost_default,
    ),
    CAR_DIMS=__user_config.get("car_dims", __car_dims_default),
    PED_DIMS=__user_config.get("ped_dims", __ped_dims_default),
    FORCE_NEW_DETECTIONS=__user_config.get(
        "force_new_detections", __force_new_detections_default
    ),
)

if CONFIG.MIN_LANDMARKS_CAR < 1 or CONFIG.MIN_LANDMARKS_PED < 1:
    raise ConfigError("Min. landmarks must be at least 1")

if CONFIG.USING_CONFIG_FILE:
    CONFIG.CONFIG_FILE = __config_file_default.as_posix()

if CONFIG.USING_MEDIAN_CLUSTER:
    CONFIG.MAD_SCALE_FACTOR = __user_config.get(
        "mad_scale_factor", __mad_scale_factor_default
    )
else:
    _default_radius = __user_config.get("cluster_radius", __cluster_radius_default)
    CONFIG.CLUSTER_RADIUS_CAR = __user_config.get("cluster_radius_car", _default_radius)
    CONFIG.CLUSTER_RADIUS_PED = __user_config.get("cluster_radius_ped", _default_radius)

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

if CONFIG.FEATURE_MATCHER == "superpoint":
    CONFIG.SUPERPOINT_WEIGHTS_PATH = __user_config.get(
        "superpoint_weights",
        (MODULE_PATH / "thirdparty" / "data" / "sp_v6").as_posix(),
    )
elif CONFIG.FEATURE_MATCHER == "superpoint_preprocessed":
    CONFIG.SUPERPOINT_PREPROCESSED_PATH = __user_config.get(
        "superpoint_preprocessed_path",
        (
            __preprocessed_path_default / "superpoint" / __kitti_scene_default.zfill(4)
        ).as_posix(),
    )


def get_config_dict() -> Dict[str, Any]:
    return CONFIG.__dict__
