"""Contains data structures used throughout bamot.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from dataclasses_json import config, dataclass_json
from marshmallow import fields

ObjectId = int
FeatureId = int
TrackId = int
TimeCamId = int


FeatureTrack = Dict[ObjectId, FeatureId]
numpy_serialization_config = config(
    encoder=lambda x: x.tolist(),
    decoder=np.array,
    mm_field=fields.List(fields.List(fields.Float())),
)


@dataclass
class StereoImage:
    left: np.ndarray
    right: np.ndarray


@dataclass_json
@dataclass
class Feature:
    u: float
    v: float
    descriptor: np.ndarray = field(metadata=numpy_serialization_config)


@dataclass_json
@dataclass
class ObjectDetection:
    convex_hull: List[Tuple[int, int]]
    track_id: Optional[TrackId] = None
    features: Optional[List[Feature]] = None


@dataclass
class Camera:
    project: Callable[[np.ndarray], np.ndarray]
    back_project: Callable[[np.ndarray], np.ndarray]


Match = Tuple[FeatureId, FeatureId]


@dataclass
class MatchData:
    matches: List[Match]


@dataclass
class FeatureMatcher:
    detect_features: Callable[
        ["FeatureMatcher", np.ndarray, Optional[np.ndarray]], List[Feature]
    ]
    match_features: Callable[
        ["FeatureMatcher", List[Feature], List[Feature]], List[Match]
    ]


@dataclass
class StereoCamera:
    left: Camera
    right: Camera
    T_left_right: np.ndarray


@dataclass
class Observation:
    descriptor: np.ndarray

    pt_2d: np.ndarray  # feature coordinates
    timecam_id: TimeCamId


@dataclass
class Landmark:
    pt_3d: np.ndarray  # w.r.t. object
    observations: List[Observation]


@dataclass
class ObjectTrack:
    landmarks: List[Landmark]
    current_pose: np.ndarray  # w.r.t. world
    velocity: np.ndarray
    poses: Dict[TimeCamId, np.ndarray]  # changing poses over time


@dataclass
class StereoObjectDetection:
    left: ObjectDetection
    right: ObjectDetection


@dataclass
class TrackMatch:
    track_index: int
    detection_index: int
