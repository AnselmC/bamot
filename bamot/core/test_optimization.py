import numpy as np
import pytest

from bamot.core.base_types import (CameraParameters, Landmark, ObjectTrack,
                                   Observation, StereoCamera)
from bamot.core.optimization import object_bundle_adjustment
from bamot.util.cv import from_homogeneous_pt, project, to_homogeneous_pt

RNG = np.random.default_rng()


@pytest.fixture
def camera_params():
    return CameraParameters(fx=500, fy=500, cx=320, cy=240)


@pytest.fixture
def T_left_right():
    T = np.identity(4)
    T[3, 3] = 0.03  # baseline of 3 cms


@pytest.fixture
def stereo_camera(camera_params, T_left_right):
    return StereoCamera(
        left=camera_params, right=camera_params, T_left_right=T_left_right
    )


@pytest.fixture
def steps():
    return 10


@pytest.fixture(scope="module")
def object_points():
    # landmarks are w.r.t. object coordinate system (which is identical to first camera seeing object)
    # sample 500 points uniformly around (x, y, z) = (1, 1, 3) with max dimensions of 3m around center
    # add landmarks
    pts_3d = RNG.uniform(-1.5, 1.5, (500, 3)) + np.array([1, 1, 3])
    return pts_3d


@pytest.fixture
def object_poses(steps):
    poses = {}
    pose = np.identity(4)
    poses[0] = pose
    motion_transform = np.identity(4)
    motion_translation = np.array([0.05, 0.01, 0.1])
    motion_transform[:3, 3] = motion_translation
    for i in range(1, steps - 1):
        pose = motion_transform @ poses[i - 1]
        poses[i] = pose
    return poses


@pytest.fixture
def landmarks(object_points, object_poses, camera_params):
    # go through all object points
    lms = {}
    for i, pt_obj in enumerate(object_points):
        # for every object pose, decide whether point was seen
        # transform point to current pose
        # project onto camera + add some noise
        observations = []
        for pose_id, pose in object_poses.items():
            if RNG.normal(0.8, 0.5) <= 0.5:
                # point isn't observed at this pose
                continue
            pt_3d = from_homogeneous_pt(pose @ to_homogeneous_pt(pt_obj))
            pt_2d = (project(camera_params, pt_3d) + RNG.random() * 2).reshape(
                (2,)
            )  # up to two pixel error
            observations.append(
                Observation(descriptor=None, pt_2d=pt_2d, timecam_id=(pose_id, 0))
            )
        landmark = Landmark(pt_obj, observations)
        lms[i] = landmark
    return lms


@pytest.fixture
def object_track(landmarks, object_poses):
    return ObjectTrack(
        landmarks=landmarks,
        current_pose=list(object_poses.values())[-1],
        poses=object_poses,
        velocity=np.array([0.0, 0.0, 0.0]),
        active=True,
    )


def test_object_bundle_adjustment(object_track, object_poses, stereo_camera):
    updated_object_track = object_bundle_adjustment(
        object_track=object_track, all_poses=object_poses, stereo_cam=stereo_camera
    )
    assert False


# TODO:
# Create same test but with example optimization and see if it converges
