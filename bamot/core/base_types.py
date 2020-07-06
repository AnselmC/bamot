"""Contains data structures used throughout bamot.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

TimeCamId = int  # uniquely identifies image (by left/right and img no.)
FeatureId = int  # feature idx for a given Object
FeatureTrack = Dict[TimeCamId, FeatureId]


@dataclass
class StereoImage:
    left: np.ndarray
    right: np.ndarray


@dataclass
class Feature:
    u: int
    v: int
    descriptor: np.ndarray


@dataclass
class ObjectDetection:
    object_mask: np.ndarray
    img_id: TimeCamId
    track_id: Optional[int] = None
    features: List[Feature] = []


@dataclass
class Camera:
    project: Callable[[np.ndarray], np.ndarray]
    back_project: Callable[[np.ndarray], np.ndarray]


@dataclass
class Match:
    FeatureIds: Tuple[FeatureId, FeatureId]


@dataclass
class MatchData:
    matches: List[Match]
    timecamids: Tuple[TimeCamId, TimeCamId]


@dataclass
class FeatureMatcher:
    detect_features: Callable[[np.ndarray], List[Feature]]
    match_features: Callable[[Tuple[List[Feature], List[Feature]]], List[Match]]


@dataclass
class StereoCamera:
    left: Camera
    right: Camera
    T_left_right: np.ndarray


@dataclass
class Landmark:
    pt_3d: np.ndarray
    observations: FeatureTrack


@dataclass
class ObjectTrack:
    landmarks: List[Landmark]
    pos_3d: np.ndarray
    velocity: np.ndarray


@dataclass
class StereoObjectDetection:
    left: ObjectDetection
    right: ObjectDetection
