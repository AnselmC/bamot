from typing import List

from bamot.core.base_types import (FeatureMatcher, ObjectTrack, StereoImage,
                                   StereoObjectDetection)
from bamot.utils.cv import get_convex_hull, project_landmarks
from hungarian_algorithm import algorithm as ha
from shapely.geometry import Polygon


def _mask_img():
    pass


def _get_features_from_landmarks():
    pass


def _add_new_landmarks():
    pass


def _get_object_associations(
    detections: List[StereoObjectDetection], object_tracks: List[ObjectTrack]
) -> Dict[Any, Any]:
    detections_by_id = {}
    tracks_by_id = {}
    graph = {}
    for i, detection in enumerate(detections):
        # compute IoU for left seg
        poly_detection = Polygon(get_convex_hull(detection.left.object_mask))
        graph[id(detection)] = {}
        detections_by_id[id(detection)] = detection
        for track in object_tracks:
            projected_landmarks = project_landmarks(track.landmarks)
            poly_track = Polygon(get_convex_hull(projected_landmarks))
            iou = (poly_detection.intersection(poly_track)) / (
                poly_detection.union(poly_track)
            )
            graph[id(detection)][id(track)] = iou
            if i == 0:
                tracks_by_id[id(track)] = track

    # get matches from hulgarian algo
    matches = ha.find_matching(graph, matching_type="max", return_type="list")
    return matches


def step(
    new_detections: List[StereoObjectDetection],
    stereo_image: StereoImage,
    object_tracks: List[ObjectTrack],
    matcher: FeatureMatcher,
) -> List[ObjectTrack]:
    if all(lambda x: x.track_id is None):
        matches = _get_object_associations(new_detections)
    else:
        pass
    # per match, match features
    for match in matches:
        detection = new_detections[match.detection_idx]
        track = object_tracks[match.track_idx]
        # first, detect features on detection
        left_obj = _mask_img(detection.left, stereo_image.left)
        right_obj = _mask_img(detection.right, stereo_image.right)
        # detect features per new detection
        left_features = matcher.detect_features(left_obj)
        right_features = matcher.detect_features(right_obj)
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
        track.landmarks = _add_new_landmarks(
            track.landmarks, track_matches, stereo_matches
        )
        # do BA (TODO: separate thread)
        # BA optimizes only landmark positions -> SLAM optimizes ego motion

    # kalman filter
    # return updated object_tracks
