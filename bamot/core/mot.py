from typing import List

import numpy as np
from shapely.geometry import Polygon

from bamot.core.base_types import (Feature, FeatureMatcher, Landmark, Match,
                                   ObjectDetection, ObjectTrack, Observation,
                                   StereoImage, StereoObjectDetection,
                                   TrackMatch)
from bamot.core.optimization import object_bundle_adjustment
from bamot.util.cv import (get_convex_hull, get_convex_hull_mask, mask_img,
                           project_landmarks)
from hungarian_algorithm import algorithm as ha


def _add_new_landmarks_and_observations(
    landmarks: List[Landmark], track_matches: List[Match], stereo_matches: List[Match]
) -> List[Landmark]:
    # TODO
    # add new observations to existing landmarks
    # add new landmarks
    for features_idx, landmark_idx in track_matches.matches.items():
        feature = features[features_idx]
        obs = Observation(descriptor=feature.descriptor)


def _get_object_associations(
    detections: List[StereoObjectDetection], object_tracks: List[ObjectTrack]
) -> List[TrackMatch]:
    graph = {}
    for i, detection in enumerate(detections):
        # compute IoU for left seg
        poly_detection = Polygon(detection.left.convex_hull)
        graph[i] = {}
        for j, track in enumerate(object_tracks):
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
    distances = np.zeros((len(observations), (observations)))
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
        features.append(Feature(u=None, v=None, descriptor=descriptor))
    return features


def step(
    new_detections: List[StereoObjectDetection],
    stereo_image: StereoImage,
    object_tracks: List[ObjectTrack],
    matcher: FeatureMatcher,
) -> List[ObjectTrack]:
    img_shape = stereo_image.left.shape
    if all(lambda x: x.track_id is None):
        matches = _get_object_associations(new_detections, object_tracks)
    else:
        matches = [
            TrackMatch(track_index=track_idx, detection_index=detection_idx)
            for detection_idx, track_idx in enumerate(
                map(lambda x: x.track_index, matches)
            )
        ]
    # per match, match features
    for match in matches:
        detection = new_detections[match.detection_index]
        track = object_tracks[match.track_index]
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
        left_features = matcher.detect_features(left_obj, mask=left_obj_mask)
        right_features = matcher.detect_features(right_obj, mask=left_obj_mask)
        detection.left.features = left_features
        detection.right.features = right_features
        # match stereo features
        stereo_matches = matcher.match_features(left_features, right_features)
        # match left features with track features
        track_matches = matcher.match_features(
            left_features, _get_features_from_landmarks(track.landmarks)
        )
        # add new feature observations from track matches
        # add new landmarks from stereo matches
        track.landmarks = _add_new_landmarks_and_observations(
            track.landmarks, track_matches, stereo_matches
        )
        # BA optimizes landmark positions w.r.t. object and object position over time
        # -> SLAM optimizes motion of camera
        # TODO: setup cameras
        track = object_bundle_adjustment(object_track=track, cameras=cameras)
        object_tracks[match.track_index] = track

    return object_tracks
