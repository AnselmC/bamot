from dataclasses import dataclass
from typing import Callable, List

import numpy as np


@dataclass
class ObjectDetection3D:
    # center of object in left camera coordinates
    # x points right w.r.t. camera
    # y points down
    # z points forward
    x: float
    y: float
    z: float
    # dimensions of object in object coordinates, i.e.
    # from center width/2 in positive & negative x-direction
    # from center height/2 in positive & negative y-direction
    # from center length/2 in positive & negative z-direction
    height: float
    width: float
    length: float
    # roration around y-axis
    ry: float


@dataclass
class Feature:
    u: int
    v: int
    descriptor: np.ndarray


@dataclass
class ExtendedObjectDetection3D:
    object_detection: ObjectDetection3D
    left_projected_bounding_box: List[np.ndarray]
    right_projected_bounding_box: List[np.ndarray]
    left_features: List[Feature]
    right_features: List[Feature]


@dataclass
class Camera:
    T_0_i: np.ndarray
    project: Callable[[np.ndarray], np.ndarray]
    back_project: Callable[[np.ndarray], np.ndarray]
