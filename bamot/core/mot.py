""" Core code for MOT with BA
"""
import logging
import queue
from typing import Dict, Iterable, List

import numpy as np
from hungarian_algorithm import algorithm as ha
from shapely.geometry import Polygon

from bamot.core.base_types import (Feature, FeatureMatcher, ImageId, Landmark,
                                   Match, ObjectTrack, Observation,
                                   StereoCamera, StereoImage,
                                   StereoObjectDetection, TrackMatch)
from bamot.core.optimization import object_bundle_adjustment
from bamot.util.cv import (back_project, from_homogeneous_pt, get_convex_hull,
                           get_convex_hull_mask, mask_img, project_landmarks,
                           to_homogeneous_pt, triangulate)

LOGGER = logging.getLogger("CORE:MOT")


def _add_new_landmarks_and_observations(
    landmarks: List[Landmark],
    track_matches: List[Match],
    stereo_matches: List[Match],
    left_features: List[Feature],
    right_features: List[Feature],
    stereo_cam: StereoCamera,
    T_cam_obj: np.ndarray,
    img_id: int,
) -> List[Landmark]:
    # add new observations to existing landmarks
    already_added_features = []
    left_timecam_id = (img_id, 0)
    right_timecam_id = (img_id, 1)
    for features_idx, landmark_idx in track_matches:
        feature = left_features[features_idx]
        already_added_features.append(features_idx)
        obs = Observation(
            descriptor=feature.descriptor,
            pt_2d=np.array([feature.u, feature.v]),
            timecam_id=left_timecam_id,
        )
        landmarks[landmark_idx].observations.append(obs)
    LOGGER.debug("Added %d observations", len(already_added_features))

    # add new landmarks
    for left_feature_idx, right_feature_idx in stereo_matches:
        # check whether landmark exists already
        if left_feature_idx in already_added_features:
            continue
        # if not, triangulate
        left_feature = left_features[left_feature_idx]
        right_feature = right_features[right_feature_idx]
        left_pt = np.array([left_feature.u, left_feature.v])
        right_pt = np.array([right_feature.u, right_feature.v])
        vec_left = back_project(stereo_cam.left, left_pt)
        vec_right = back_project(stereo_cam.right, right_pt)
        R_left_right = stereo_cam.T_left_right[:3, :3]
        t_left_right = stereo_cam.T_left_right[:3, 3].reshape(3, 1)
        pt_3d_left_cam = triangulate(vec_left, vec_right, R_left_right, t_left_right)
        if pt_3d_left_cam[-1] < 0:
            # triangulated point should not be behind camera
            continue
        pt_3d = from_homogeneous_pt(
            np.linalg.inv(T_cam_obj) @ to_homogeneous_pt(pt_3d_left_cam)
        )
        # create new landmark
        left_obs = Observation(
            descriptor=left_feature.descriptor,
            pt_2d=left_pt,
            timecam_id=left_timecam_id,
        )
        right_obs = Observation(
            descriptor=right_feature.descriptor,
            pt_2d=right_pt,
            timecam_id=right_timecam_id,
        )

        landmark = Landmark(pt_3d, [left_obs, right_obs])
        landmarks.append(landmark)

    LOGGER.debug("Created %d landmarks", len(landmarks))
    return landmarks


def _get_object_associations(
    detections: List[StereoObjectDetection], object_tracks: Dict[int, ObjectTrack]
) -> List[TrackMatch]:
    graph: Dict[int, Dict[int, float]] = {}
    # TODO: fix
    for i, detection in enumerate(detections):
        # compute IoU for left seg
        poly_detection = Polygon(detection.left.convex_hull)
        graph[i] = {}
        for j, track in object_tracks.items():
            projected_landmarks = project_landmarks(track.landmarks)
            poly_track = Polygon(get_convex_hull(projected_landmarks))
            iou = (poly_detection.intersection(poly_track)) / (
                poly_detection.union(poly_track)
            )
            graph[i][j] = iou

    # get matches from hulgarian algo
    matches = ha.find_matching(graph, matching_type="max", return_type="list")
    track_matches = []
    for match in matches:
        detection_idx = match[0][0]
        track_idx = match[0][1]
        track_match = TrackMatch(track_index=track_idx, detection_index=detection_idx)
        track_matches.append(track_match)
    return track_matches


def _compute_median_descriptor(
    observations: List[Observation], norm: int
) -> np.ndarray:
    distances = np.zeros((len(observations), len(observations)))
    for i, obs in enumerate(observations):
        for j in range(i, len(observations)):
            other_obs = observations[j]
            # calculate distance between i and j
            dist = np.linalg.norm(obs.descriptor - other_obs.descriptor, ord=norm)
            # do for all combinations
            distances[i, j] = dist
            distances[j, i] = dist
    best_median = None
    best_idx = 0
    for i, obs in enumerate(observations):
        dist_per_descriptor = distances[i]
        median = np.median(dist_per_descriptor)
        if not best_median or median < best_median:
            best_median = median
            best_idx = 1
    return observations[best_idx].descriptor


def _get_features_from_landmarks(landmarks: List[Landmark]) -> List[Feature]:
    # todo: refactor
    features = []
    for landmark in landmarks:
        obs = landmark.observations
        descriptor = _compute_median_descriptor(obs, norm=2)
        features.append(Feature(u=0.0, v=0.0, descriptor=descriptor))
    return features


def run(
    images: Iterable[StereoImage],
    detections: Iterable[List[StereoObjectDetection]],
    feature_matcher: FeatureMatcher,
    stereo_cam: StereoCamera,
    slam_data: queue.Queue,
    shared_data: queue.Queue,
):
    object_tracks: Dict[int, ObjectTrack] = {}
    LOGGER.info("Starting MOT run")

    for it, (stereo_image, new_detections) in enumerate(zip(images, detections)):
        all_poses = slam_data.get()
        current_pose = all_poses[it]
        object_tracks = step(
            new_detections=new_detections,
            stereo_image=stereo_image,
            object_tracks=object_tracks,
            matcher=feature_matcher,
            stereo_cam=stereo_cam,
            img_id=it,
            current_cam_pose=current_pose,
            all_poses=all_poses,
        )
        shared_data.put({"object_tracks": object_tracks})


def step(
    new_detections: List[StereoObjectDetection],
    stereo_image: StereoImage,
    object_tracks: Dict[int, ObjectTrack],
    matcher: FeatureMatcher,
    stereo_cam: StereoCamera,
    all_poses: Dict[ImageId, np.ndarray],
    img_id: ImageId,
    current_cam_pose: np.ndarray,
) -> Dict[int, ObjectTrack]:
    img_shape = stereo_image.left.shape
    LOGGER.debug("Running step for image %d", img_id)
    if all(map(lambda x: x.left.track_id is None, new_detections)):
        # no track ids yet
        matches = _get_object_associations(new_detections, object_tracks)
    else:
        # track ids already exist
        matches = [
            TrackMatch(track_index=track_idx, detection_index=detection_idx)
            for detection_idx, track_idx in enumerate(
                map(lambda x: x.left.track_id, new_detections)
            )
        ]
        # if new track ids are present, the tracks need to be added to the object_tracks
        for match in matches:
            if object_tracks.get(match.track_index) is None:
                object_tracks[match.track_index] = ObjectTrack(
                    landmarks=[],
                    current_pose=current_cam_pose,
                    velocity=np.array([0.0, 0.0, 0.0]),
                    poses={img_id: current_cam_pose},
                )
    # per match, match features
    active_tracks = []
    matched_detections = []
    LOGGER.debug("%d matches with object tracks", len(matches))
    for match in matches:
        detection = new_detections[match.detection_index]
        track = object_tracks[match.track_index]
        active_tracks.append(match.track_index)
        matched_detections.append(match.detection_index)
        # mask out object from image
        left_obj_mask = get_convex_hull_mask(
            np.array(detection.left.convex_hull), img_shape=img_shape
        )
        right_obj_mask = get_convex_hull_mask(
            np.array(detection.right.convex_hull), img_shape=img_shape
        )
        # left_obj = mask_img(left_obj_mask, stereo_image.left, dilate=True)
        # right_obj = mask_img(right_obj_mask, stereo_image.right, dilate=True)
        left_obj = stereo_image.left
        right_obj = stereo_image.right
        # detect features per new detection
        left_features = matcher.detect_features(left_obj, left_obj_mask)
        LOGGER.debug("Detected %d features on left object", len(left_features))
        right_features = matcher.detect_features(right_obj, left_obj_mask)
        LOGGER.debug("Detected %d features on right object", len(right_features))
        detection.left.features = left_features
        detection.right.features = right_features
        # match stereo features
        stereo_matches = matcher.match_features(left_features, right_features)
        LOGGER.debug("%d stereo matches", len(stereo_matches))
        # match left features with track features
        track_matches = matcher.match_features(
            left_features, _get_features_from_landmarks(track.landmarks)
        )
        LOGGER.debug("%d track matches", len(track_matches))
        # add new feature observations from track matches
        # add new landmarks from stereo matches
        T_world_obj = track.current_pose
        T_cam_obj = np.linalg.inv(current_cam_pose) @ T_world_obj
        track.landmarks = _add_new_landmarks_and_observations(
            landmarks=track.landmarks,
            track_matches=track_matches,
            stereo_matches=stereo_matches,
            left_features=left_features,
            right_features=right_features,
            stereo_cam=stereo_cam,
            img_id=img_id,
            T_cam_obj=T_cam_obj,
        )
        # BA optimizes landmark positions w.r.t. object and object position over time
        # -> SLAM optimizes motion of camera
        # cameras maps a timecam_id (i.e. frame + left/right) to a camera pose and camera parameters
        if len(track.poses) > 1:
            track = object_bundle_adjustment(
                object_track=track, all_poses=all_poses, stereo_cam=stereo_cam
            )
            object_tracks[match.track_index] = track
    # Set old tracks inactive
    old_tracks = set(range(len(object_tracks))).difference(set(active_tracks))
    LOGGER.debug("Deactivating %d tracks", len(old_tracks))
    for track_id in old_tracks:
        object_tracks[track_id].active = False
    # add new tracks
    new_tracks = set(range(len(new_detections))).difference(set(matched_detections))
    LOGGER.debug("Adding %d new tracks", len(new_tracks))
    for detection_id in new_tracks:
        detection = new_detections[detection_id]
        track_id = max(object_tracks.keys()) + 1
        # mask out object from image
        left_obj_mask = get_convex_hull_mask(
            detection.left.convex_hull, img_shape=img_shape
        )
        right_obj_mask = get_convex_hull_mask(
            detection.right.convex_hull, img_shape=img_shape
        )
        left_obj = mask_img(left_obj_mask, stereo_image.left, dilate=True)
        right_obj = mask_img(right_obj_mask, stereo_image.right, dilate=True)
        # detect features per new detection
        left_features = matcher.detect_features(left_obj, left_obj_mask)
        right_features = matcher.detect_features(right_obj, left_obj_mask)
        detection.left.features = left_features
        detection.right.features = right_features
        # match stereo features
        stereo_matches = matcher.match_features(left_features, right_features)
        # match left features with track features
        velocity = np.array([0.0, 0.0, 0.0])
        # initial pose is camera pose
        current_pose = current_cam_pose
        poses = {img_id: current_pose}
        # initial landmarks are triangulated stereo matches
        landmarks = _add_new_landmarks_and_observations(
            landmarks=[],
            track_matches=[],
            stereo_matches=stereo_matches,
            left_features=left_features,
            right_features=right_features,
            stereo_cam=stereo_cam,
            T_cam_obj=np.identity(4),
            img_id=img_id,
        )
        obj_track = ObjectTrack(
            landmarks=landmarks,
            current_pose=current_pose,
            velocity=velocity,
            poses=poses,
        )
        object_tracks[track_id] = obj_track

    LOGGER.debug("Finished step")
    LOGGER.debug("=" * 90)
    return object_tracks
