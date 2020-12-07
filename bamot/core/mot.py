""" Core code for BAMOT
"""
import copy
import logging
import queue
import time
import uuid
from threading import Event
from typing import Dict, Iterable, List, Tuple

import cv2
import g2o
import numpy as np
import pathos
from bamot.config import CONFIG as config
from bamot.core.base_types import (CameraParameters, Feature, FeatureMatcher,
                                   ImageId, Landmark, Match, ObjectTrack,
                                   Observation, StereoCamera, StereoImage,
                                   StereoObjectDetection, TrackId, TrackMatch,
                                   get_camera_parameters_matrix)
from bamot.core.optimization import object_bundle_adjustment
from bamot.util.cv import (back_project, from_homogeneous,
                           get_center_of_landmarks, get_feature_matcher,
                           to_homogeneous, triangulate)
from bamot.util.misc import get_mad, timer

LOGGER = logging.getLogger("CORE:MOT")


def _remove_outlier_landmarks(landmarks, matched_landmarks, cls, track_logger):
    if matched_landmarks:
        landmarks_to_remove = []
        points = np.array(matched_landmarks)

        # for landmark in landmarks.values():
        #    points.append(landmark.pt_3d)
        # points = np.array(points)
        cluster_median_center = np.median(points, axis=0)
        for lid, lm in landmarks.items():
            if not config.USING_MEDIAN_CLUSTER:
                cluster_size = (
                    config.CLUSTER_SIZE_CAR if cls == "car" else config.CLUSTER_SIZE_PED
                )
                if np.linalg.norm(lm.pt_3d - cluster_median_center) > (
                    cluster_size / 2
                ):
                    landmarks_to_remove.append(lid)
            else:
                if np.linalg.norm(
                    lm.pt_3d - cluster_median_center
                ) > config.MAD_SCALE_FACTOR * get_mad(points):
                    landmarks_to_remove.append(lid)

        track_logger.debug("Removing %d outlier landmarks", len(landmarks_to_remove))
        for lid in landmarks_to_remove:
            landmarks.pop(lid)


def get_median_translation(object_track):
    translations = []
    frames = list(object_track.poses.keys())
    for i in range(len(frames[-2 * config.SLIDING_WINDOW_BA :]) - 1):
        img_id_0 = frames[i]
        img_id_1 = frames[i + 1]
        pose0 = object_track.poses[img_id_0]
        pose1 = object_track.poses[img_id_1]
        translations.append(np.linalg.norm((np.linalg.inv(pose0) @ pose1)[:3, 3]))
    return np.median(translations)


def _valid_motion(Tr_rel, obj_cls, badly_tracked_frames):
    curr_translation = np.linalg.norm(Tr_rel[:3, 3])
    max_speed = config.MAX_SPEED_CAR if obj_cls == "car" else config.MAX_SPEED_PED
    LOGGER.debug("Current translation: %.2f", float(curr_translation))
    LOGGER.debug("Max. allowed translation: %.2f", max_speed / config.FRAME_RATE)
    valid_motion = curr_translation < (
        (badly_tracked_frames + 1) * 0.5 * (max_speed / config.FRAME_RATE)
    )
    LOGGER.debug("Valid motion: %s", valid_motion)
    return valid_motion


def _localize_object(
    left_features: List[Feature],
    track_matches: List[Match],
    landmark_mapping: Dict[int, int],
    landmarks: Dict[int, Landmark],
    T_cam_obj: np.ndarray,
    camera_params: CameraParameters,
    logger: logging.Logger,
    num_iterations: int = 2000,
    reprojection_error: float = 1.0,
) -> Tuple[np.ndarray, bool]:
    pts_3d = []
    pts_2d = []

    logger.debug(
        "Localizing object based on %d point correspondences", len(track_matches)
    )
    # build pt arrays
    for features_idx, landmark_idx in track_matches:
        pt_3d = landmarks[landmark_mapping[landmark_idx]].pt_3d
        feature = left_features[features_idx]
        pt_2d = np.array([feature.u, feature.v])
        pts_3d.append(pt_3d)
        pts_2d.append(pt_2d)
    pts_3d = np.array(pts_3d).reshape(-1, 3)
    pts_2d = np.array(pts_2d).reshape(-1, 2)
    rot = T_cam_obj[:3, :3]
    # use previous pose + constant motion as initial guess
    trans = T_cam_obj[:3, 3]
    # solvePnPRansac estimates object pose, not camera pose
    successful, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=pts_3d,
        imagePoints=pts_2d,
        cameraMatrix=get_camera_parameters_matrix(camera_params),
        distCoeffs=None,
        rvec=cv2.Rodrigues(rot)[0],
        tvec=trans.astype(float),
        useExtrinsicGuess=True,
        iterationsCount=num_iterations,
        reprojectionError=reprojection_error,
    )
    num_inliers = len(inliers) if inliers is not None else 0
    inlier_ratio = num_inliers / len(track_matches)
    logger.debug("Inlier ratio for PnP: %.2f", inlier_ratio)
    if successful and inlier_ratio > 0.25:
        logger.debug("Optimization successful! Found %d inliers", len(inliers))
        logger.debug("Running optimization with inliers...")
        successful, rvec, tvec = cv2.solvePnP(
            objectPoints=np.array([mp for i, mp in enumerate(pts_3d) if i in inliers]),
            imagePoints=np.array([ip for i, ip in enumerate(pts_2d) if i in inliers]),
            cameraMatrix=get_camera_parameters_matrix(camera_params),
            distCoeffs=None,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
        )
        if successful:
            LOGGER.debug("Inlier optimization successful!")
            rot, _ = cv2.Rodrigues(rvec)
            optimized_pose = np.identity(4)
            optimized_pose[:3, :3] = rot
            optimized_pose[:3, 3] = tvec
            LOGGER.debug("Optimized pose from \n%s\nto\n%s", T_cam_obj, optimized_pose)
            return optimized_pose, True
    logger.debug("Optimization failed...")
    return T_cam_obj, False


def _add_new_landmarks_and_observations(
    landmarks: Dict[int, Landmark],
    track_matches: List[Match],
    landmark_mapping: Dict[int, int],
    stereo_matches: List[Match],
    left_features: List[Feature],
    right_features: List[Feature],
    stereo_cam: StereoCamera,
    T_cam_obj: np.ndarray,
    img_id: int,
    logger: logging.Logger,
) -> Dict[int, Landmark]:
    already_added_features = []
    stereo_match_dict = {}
    for left_feature_idx, right_feature_idx in stereo_matches:
        stereo_match_dict[left_feature_idx] = right_feature_idx

    matched_landmarks = []

    # add new observations to existing landmarks
    for features_idx, landmark_idx in track_matches:
        feature = left_features[features_idx]
        pt_obj = landmarks[landmark_mapping[landmark_idx]].pt_3d
        pt_cam = from_homogeneous(T_cam_obj @ to_homogeneous(pt_obj))
        z = pt_cam[2]
        if (
            z < 0.5 or np.linalg.norm(pt_cam) > config.MAX_DIST
        ):  # don't add landmarks that are very behind camera/very close or far away
            continue
        # stereo observation
        if stereo_match_dict.get(features_idx) is not None:
            right_feature = right_features[stereo_match_dict[features_idx]]
            # check epipolar constraint
            if np.allclose(feature.v, right_feature.v, atol=1):
                feature_pt = np.array([feature.u, feature.v, right_feature.u])
            else:
                feature_pt = np.array([feature.u, feature.v])

        # mono observation
        else:
            feature_pt = np.array([feature.u, feature.v])
        obs = Observation(
            descriptor=feature.descriptor, pt_2d=feature_pt, img_id=img_id
        )
        matched_landmarks.append(pt_obj)
        already_added_features.append(features_idx)
        landmarks[landmark_mapping[landmark_idx]].observations.append(obs)
    logger.debug("Added %d observations", len(already_added_features))

    # add new landmarks
    created_landmarks = 0
    bad_matches = []
    for left_feature_idx, right_feature_idx in stereo_matches:
        landmark_id = uuid.uuid1().int
        # check whether landmark exists already
        if left_feature_idx in already_added_features:
            continue
        # if not, triangulate
        left_feature = left_features[left_feature_idx]
        right_feature = right_features[right_feature_idx]
        left_pt = np.array([left_feature.u, left_feature.v])
        right_pt = np.array([right_feature.u, right_feature.v])
        feature_pt = np.array([left_feature.u, left_feature.v, right_feature.u])
        if not np.allclose(left_feature.v, right_feature.v, atol=1):
            # match doesn't fullfill epipolar constraint
            bad_matches.append((left_feature_idx, right_feature_idx))
            continue
        vec_left = back_project(stereo_cam.left, left_pt)
        vec_right = back_project(stereo_cam.right, right_pt)
        R_left_right = stereo_cam.T_left_right[:3, :3]
        t_left_right = stereo_cam.T_left_right[:3, 3].reshape(3, 1)
        try:
            pt_3d_left_cam = triangulate(
                vec_left, vec_right, R_left_right, t_left_right
            )
        except np.linalg.LinAlgError:
            bad_matches.append((left_feature_idx, right_feature_idx))
            continue
        if pt_3d_left_cam[-1] < 0.5 or np.linalg.norm(pt_3d_left_cam) > config.MAX_DIST:
            # triangulated point should not be behind camera (or very close) or too far away
            bad_matches.append((left_feature_idx, right_feature_idx))
            continue
        pt_3d_obj = from_homogeneous(
            np.linalg.inv(T_cam_obj) @ to_homogeneous(pt_3d_left_cam)
        ).reshape(3, 1)
        # create new landmark
        obs = Observation(
            descriptor=left_feature.descriptor, pt_2d=feature_pt, img_id=img_id
        )
        matched_landmarks.append(pt_3d_obj)
        landmark = Landmark(pt_3d_obj, [obs])
        landmarks[landmark_id] = landmark
        created_landmarks += 1

    for match in bad_matches:
        stereo_matches.remove(match)
    logger.debug("Created %d landmarks", created_landmarks)
    return landmarks, matched_landmarks


def _get_median_descriptor(observations: List[Observation], norm: int,) -> np.ndarray:
    subset = observations[-config.SLIDING_WINDOW_DESCRIPTORS :]
    distances = np.zeros((len(subset), len(subset)))
    for i, obs in enumerate(subset):
        for j in range(i, len(subset)):
            other_obs = subset[j]
            # calculate distance between i and j
            dist = np.linalg.norm(obs.descriptor - other_obs.descriptor, ord=norm)
            # do for all combinations
            distances[i, j] = dist
            distances[j, i] = dist
    best_median = None
    best_idx = 0
    for i, obs in enumerate(subset):
        dist_per_descriptor = distances[i]
        median = np.median(dist_per_descriptor)
        if not best_median or median < best_median:
            best_median = median
            best_idx = i
    return subset[best_idx].descriptor


def _get_features_from_landmarks(
    landmarks: Dict[int, Landmark]
) -> Tuple[List[Feature], Dict[int, int]]:
    features = []
    landmark_mapping = {}
    idx = 0
    for lid, landmark in landmarks.items():
        obs = landmark.observations
        descriptor = _get_median_descriptor(obs, norm=2)
        features.append(Feature(u=0.0, v=0.0, descriptor=descriptor))
        landmark_mapping[idx] = lid
        idx += 1
    return features, landmark_mapping


@timer
def run(
    images: Iterable[StereoImage],
    detections: Iterable[List[StereoObjectDetection]],
    stereo_cam: StereoCamera,
    slam_data: queue.Queue,
    shared_data: queue.Queue,
    returned_data: queue.Queue,
    stop_flag: Event,
    next_step: Event,
    continuous: bool,
):
    object_tracks: Dict[int, ObjectTrack] = {}
    ba_slots: Tuple[set] = tuple(set() for _ in range(config.BA_EVERY_N_STEPS))
    LOGGER.info("Starting MOT run")

    def _process_match(
        track,
        detection,
        all_poses,
        track_id,
        stereo_cam,
        img_id,
        stereo_image,
        current_cam_pose,
        run_ba,
    ):
        track.active = True
        track_logger = logging.getLogger(f"CORE:MOT:{track_id}")
        track_logger.debug("Image: %d", img_id)
        feature_matcher = get_feature_matcher()
        # mask out object from image
        if not detection.left.features:
            left_features = feature_matcher.detect_features(
                stereo_image.left, detection.left.mask, img_id, track_id, "left"
            )
            track_logger.debug(
                "Detected %d features on left object", len(left_features)
            )
            detection.left.features = left_features
        else:
            left_features = detection.left.features
        if not detection.right.features:
            right_features = feature_matcher.detect_features(
                stereo_image.right, detection.right.mask, img_id, track_id, "right"
            )
            track_logger.debug(
                "Detected %d features on right object", len(right_features)
            )
            detection.right.features = right_features
        else:
            right_features = detection.right.features
        # match stereo features
        stereo_matches = feature_matcher.match_features(left_features, right_features)
        track_logger.debug("%d stereo matches", len(stereo_matches))
        # match left features with track features
        features, lm_mapping = _get_features_from_landmarks(track.landmarks)
        track_matches = feature_matcher.match_features(left_features, features)
        track_logger.debug("%d track matches", len(track_matches))
        # localize object
        T_world_obj = _estimate_next_pose(track, img_id)
        T_world_cam = current_cam_pose
        T_cam_obj = np.linalg.inv(T_world_cam) @ T_world_obj
        enough_track_matches = len(track_matches) >= 5
        enough_stereo_matches = len(stereo_matches) > min(len(track_matches), 10)
        successful = True
        valid_motion = True
        median_translation = get_median_translation(track)
        if enough_track_matches:
            T_cam_obj_pnp, successful = _localize_object(
                left_features=left_features,
                track_matches=track_matches,
                landmark_mapping=lm_mapping,
                landmarks=copy.deepcopy(track.landmarks),
                T_cam_obj=T_cam_obj.copy(),
                camera_params=stereo_cam.left,
                logger=track_logger,
            )

            if successful:
                if len(track.poses) >= 2:
                    T_world_obj_prev = track.poses[list(track.poses.keys())[-2]]
                    T_world_obj_pnp = T_world_cam @ T_cam_obj_pnp
                    T_rel = np.linalg.inv(T_world_obj_prev) @ T_world_obj_pnp
                    valid_motion = _valid_motion(
                        T_rel, track.cls, track.badly_tracked_frames
                    )
                    LOGGER.debug("Median translation: %.2f", median_translation)
                    if valid_motion:
                        T_cam_obj = T_cam_obj_pnp
        if not (enough_track_matches and successful and valid_motion):
            # if track_matches:
            #    track_logger.debug("Clearing track matches")
            #    track_matches.clear()
            track_logger.debug("Enough matches: %s", enough_track_matches)
            if enough_track_matches:
                track_logger.debug("PnP successful: %s", successful)
                if successful:
                    track_logger.debug("Valid motion: %s", valid_motion)
            if (
                track.badly_tracked_frames > config.MAX_BAD_FRAMES
                and enough_stereo_matches
            ):
                track_logger.debug("Clearing landmarks")
                track.landmarks.clear()
                track_matches.clear()
                track.badly_tracked_frames = 0
            else:
                track.badly_tracked_frames += 1

        T_world_obj = T_world_cam @ T_cam_obj
        # add new landmark observations from track matches
        # add new landmarks from stereo matches
        track.landmarks, matched_landmarks = _add_new_landmarks_and_observations(
            landmarks=copy.deepcopy(track.landmarks),
            track_matches=track_matches,
            landmark_mapping=lm_mapping,
            stereo_matches=stereo_matches,
            left_features=left_features,
            right_features=right_features,
            stereo_cam=stereo_cam,
            img_id=img_id,
            T_cam_obj=T_cam_obj,
            logger=track_logger,
        )
        # BA optimizes landmark positions w.r.t. object and object position over time
        # -> SLAM optimizes motion of camera
        # cameras maps a timecam_id (i.e. frame + left/right) to a camera pose and camera parameters
        if len(track.poses) > 3 and len(track.landmarks) > 0 and run_ba:
            track_logger.debug("Running BA")
            track = object_bundle_adjustment(
                object_track=copy.deepcopy(track),
                all_poses=all_poses,
                stereo_cam=stereo_cam,
                median_translation=median_translation,
            )
        # remove outlier landmarks
        _remove_outlier_landmarks(
            track.landmarks, matched_landmarks, track.cls, track_logger
        )
        # settings min_landmarks to 0 disables robust initialization
        if (
            len(track.poses) == 1
            and config.MIN_LANDMARKS
            and len(track.landmarks) < config.MIN_LANDMARKS
        ):
            track.active = False
        if track.landmarks:
            track.poses[img_id] = T_world_obj
            track.pcl_centers[img_id] = get_center_of_landmarks(
                track.landmarks.values()
            )
            track.locations[img_id] = from_homogeneous(
                track.poses[img_id]
                @ to_homogeneous(get_center_of_landmarks(track.landmarks.values()))
            )
        return track, left_features, right_features, stereo_matches

    point_cloud_sizes = {}
    for (img_id, stereo_image), new_detections in zip(images, detections):
        if config.TRACK_POINT_CLOUD_SIZES:
            for track_id, obj in object_tracks.items():
                point_cloud_size = len(obj.landmarks)
                if point_cloud_sizes.get(track_id):
                    point_cloud_sizes[track_id].append(point_cloud_size)
                else:
                    point_cloud_sizes[track_id] = [point_cloud_size]
        if stop_flag.is_set():
            break
        while not continuous and not next_step.is_set():
            time.sleep(0.05)
        next_step.clear()
        all_poses = slam_data.get()
        slam_data.task_done()
        current_pose = all_poses[img_id]
        track_ids = [obj.left.track_id for obj in new_detections]

        # clear slots
        slot_sizes = {}
        for idx, slot in enumerate(ba_slots):
            slot.clear()
            slot_sizes[idx] = 0
        # add track_ids to ba slots
        for track_id in track_ids:
            slot_idx, _ = sorted(
                [(idx, size) for idx, size in slot_sizes.items()], key=lambda x: x[1],
            )[0]
            ba_slots[slot_idx].add(track_id)
            slot_sizes[slot_idx] += 1
        tracks_to_run_ba = ba_slots[img_id % config.BA_EVERY_N_STEPS]
        LOGGER.debug("BA slots: %s", ba_slots)
        try:
            (
                object_tracks,
                all_left_features,
                all_right_features,
                all_stereo_matches,
            ) = step(
                new_detections=new_detections,
                stereo_image=stereo_image,
                object_tracks=copy.deepcopy(object_tracks),
                process_match=_process_match,
                stereo_cam=stereo_cam,
                img_id=img_id,
                current_cam_pose=current_pose,
                all_poses=all_poses,
                tracks_to_run_ba=tracks_to_run_ba,
            )
        except Exception as exc:
            LOGGER.exception("Unexpected error: %s", exc)
            break
        shared_data.put(
            {
                "object_tracks": copy.deepcopy(object_tracks),
                "stereo_image": stereo_image,
                "all_left_features": all_left_features,
                "all_right_features": all_right_features,
                "all_stereo_matches": all_stereo_matches,
                "img_id": img_id,
                "current_cam_pose": current_pose,
            }
        )
    stop_flag.set()
    shared_data.put({})
    returned_data.put(
        dict(
            trajectories=_compute_estimated_trajectories(object_tracks, all_poses),
            point_cloud_sizes=point_cloud_sizes,
        ),
    )


def _estimate_next_pose(track: ObjectTrack, img_id: ImageId) -> np.ndarray:
    available_poses = list(track.poses.keys())  # sorted by default
    # return track.poses[available_poses[-1]]
    if len(available_poses) >= 2:
        num_frames = min(int(config.SLIDING_WINDOW_BA), len(track.poses))
        T_world_obj1 = g2o.Isometry3d(track.poses[available_poses[-1]])
        LOGGER.debug("Previous pose:\n%s", T_world_obj1.matrix())
        T_world_obj0 = g2o.Isometry3d(track.poses[available_poses[-num_frames]])
        rel_translation = (T_world_obj1.translation() - T_world_obj0.translation()) / (
            num_frames
        )
        LOGGER.debug("Relative translation:\n%s", rel_translation)
        T_world_new = g2o.Isometry3d(
            T_world_obj1.rotation(), T_world_obj1.translation() + 1 * rel_translation
        )
        LOGGER.debug("Estimated new pose:\n%s", T_world_new.matrix())
        return T_world_new.matrix()
    else:
        return track.poses[available_poses[-1]]


@timer
def step(
    new_detections: List[StereoObjectDetection],
    stereo_image: StereoImage,
    object_tracks: Dict[int, ObjectTrack],
    process_match: FeatureMatcher,
    stereo_cam: StereoCamera,
    all_poses: Dict[ImageId, np.ndarray],
    img_id: ImageId,
    current_cam_pose: np.ndarray,
    tracks_to_run_ba: List[TrackId],
) -> Tuple[
    Dict[int, ObjectTrack], List[List[Feature]], List[List[Feature]], List[List[Match]]
]:
    all_left_features = []
    all_right_features = []
    all_stereo_matches = []
    LOGGER.debug("Running step for image %d", img_id)
    LOGGER.debug("Current ego pose:\n%s", current_cam_pose)
    matches = [
        TrackMatch(track_index=track_idx, detection_index=detection_idx)
        for detection_idx, track_idx in enumerate(
            map(lambda x: x.left.track_id, new_detections)
        )
    ]
    unmatched_tracks = set(object_tracks.keys()).difference(
        set([x.left.track_id for x in new_detections])
    )
    # if new track ids are present, the tracks need to be added to the object_tracks
    for match in matches:
        if object_tracks.get(match.track_index) is None:
            LOGGER.debug("Added track with index %d", match.track_index)
            object_tracks[match.track_index] = ObjectTrack(
                cls=new_detections[match.detection_index].left.cls,
                poses={img_id: current_cam_pose},
            )
    # per match, match features
    active_tracks = []
    matched_detections = []
    LOGGER.debug("%d matches with object tracks", len(matches))

    def _add_constant_motion_to_track(track: ObjectTrack, img_id: ImageId):
        if not track.poses:
            return track
        T_world_obj = _estimate_next_pose(track, img_id)
        if track.landmarks:
            track.poses[img_id] = T_world_obj
            track.pcl_centers[img_id] = get_center_of_landmarks(
                track.landmarks.values()
            )
            track.locations[img_id] = from_homogeneous(
                track.poses[img_id]
                @ to_homogeneous(get_center_of_landmarks(track.landmarks.values()))
            )
        return track

    # TODO: currently disabled bc slower than single threaded and single process
    with pathos.threading.ThreadPool(nodes=len(matches) if False else 1) as executor:
        matched_futures_to_track_index = {}
        unmatched_futures_to_track_index = {}
        for track_id in unmatched_tracks:
            track = object_tracks[track_id]
            track.fully_visible = False
            unmatched_futures_to_track_index[
                executor.apipe(
                    _add_constant_motion_to_track, track=track, img_id=img_id
                )
            ] = track_id
        for match in matches:
            detection = new_detections[match.detection_index]
            track = object_tracks[match.track_index]
            track.last_seen = img_id
            track.fully_visible = new_detections[match.detection_index]
            run_ba = match.track_index in tracks_to_run_ba
            active_tracks.append(match.track_index)
            matched_detections.append(match.detection_index)
            matched_futures_to_track_index[
                executor.apipe(
                    process_match,
                    track=track,
                    detection=detection,
                    all_poses=all_poses,
                    track_id=match.track_index,
                    stereo_cam=stereo_cam,
                    img_id=img_id,
                    stereo_image=stereo_image,
                    current_cam_pose=current_cam_pose,
                    run_ba=run_ba,
                )
            ] = match.track_index

        for future, track_index in unmatched_futures_to_track_index.items():
            track = future.get()
            object_tracks[track_index] = track

        for future, track_index in matched_futures_to_track_index.items():
            track, left_features, right_features, stereo_matches = future.get()
            object_tracks[track_index] = track
            all_left_features.append(left_features)
            all_right_features.append(right_features)
            all_stereo_matches.append(stereo_matches)

    # Set old tracks inactive
    old_tracks = set(object_tracks.keys()).difference(set(active_tracks))
    num_deactivated = 0
    for track_id in old_tracks:
        track = object_tracks[track_id]
        if (img_id - track.last_seen) > config.KEEP_TRACK_FOR_N_FRAMES_AFTER_LOST:
            track.active = False
            num_deactivated += 1
    LOGGER.debug("Deactivated %d tracks", num_deactivated)
    LOGGER.debug("Finished step %d", img_id)
    LOGGER.debug("=" * 90)
    return object_tracks, all_left_features, all_right_features, all_stereo_matches


def _compute_estimated_trajectories(
    object_tracks: Dict[int, ObjectTrack], all_poses: Dict[int, np.ndarray]
) -> Tuple[Dict[int, Dict[int, Tuple[float, float, float]]]]:
    offline_trajectories_world = {}
    offline_trajectories_cam = {}
    online_trajectories_world = {}
    online_trajectories_cam = {}
    for track_id, track in object_tracks.items():
        offline_trajectory_world = {}
        offline_trajectory_cam = {}
        online_trajectory_world = {}
        online_trajectory_cam = {}
        for img_id, pose_world_obj in track.poses.items():
            Tr_world_cam = all_poses[img_id]
            object_center = track.pcl_centers.get(img_id)
            if object_center is None:
                continue
            object_center_world_offline = pose_world_obj @ to_homogeneous(object_center)
            object_center_cam_offline = (
                np.linalg.inv(Tr_world_cam) @ object_center_world_offline
            )
            offline_trajectory_world[int(img_id)] = tuple(
                from_homogeneous(object_center_world_offline).tolist()
            )
            offline_trajectory_cam[int(img_id)] = tuple(
                from_homogeneous(object_center_cam_offline).tolist()
            )
        for img_id, object_center_world_online in track.locations.items():
            Tr_world_cam = all_poses[img_id]
            object_center_cam_online = np.linalg.inv(Tr_world_cam) @ to_homogeneous(
                object_center_world_online
            )
            online_trajectory_world[int(img_id)] = tuple(
                object_center_world_online.tolist()
            )
            online_trajectory_cam[int(img_id)] = tuple(
                from_homogeneous(object_center_cam_online).tolist()
            )
        offline_trajectories_world[int(track_id)] = offline_trajectory_world
        offline_trajectories_cam[int(track_id)] = offline_trajectory_cam
        online_trajectories_world[int(track_id)] = online_trajectory_world
        online_trajectories_cam[int(track_id)] = online_trajectory_cam
    return (
        (offline_trajectories_world, offline_trajectories_cam),
        (online_trajectories_world, online_trajectories_cam),
    )
