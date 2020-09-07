import logging
import queue
from threading import Event
from typing import Dict, List, Tuple

import cv2
import numpy as np
import open3d as o3d

from bamot.core.base_types import Feature, Match, ObjectTrack, StereoImage
from bamot.util.cv import (from_homogeneous_pt, get_center_of_landmarks,
                           to_homogeneous_pt)

LOGGER = logging.getLogger("UTIL:VIEWER")
LOGGER.setLevel(logging.ERROR)
RNG = np.random.default_rng()
Color = np.ndarray
# Create some random colors
COLORS: List[Color] = RNG.random((42, 3))


def _draw_features(img: np.ndarray, features: List[Feature]) -> np.ndarray:
    keypoints = [cv2.KeyPoint(x=f.u, y=f.v, _size=1) for f in features]
    return cv2.drawKeypoints(img, keypoints, None)


def _draw_matches(
    left_img: np.ndarray,
    left_features: List[Feature],
    right_img: np.ndarray,
    right_features: List[Feature],
    stereo_matches: List[Match],
) -> StereoImage:
    left_keypoints = [cv2.KeyPoint(x=f.u, y=f.v, _size=1) for f in left_features]
    right_keypoints = [cv2.KeyPoint(x=f.u, y=f.v, _size=1) for f in right_features]
    # left is train, right is query?
    matches = [
        cv2.DMatch(_queryIdx=leftIdx, _trainIdx=rightIdx, _imgIdx=0, _distance=0)
        for leftIdx, rightIdx in stereo_matches
    ]
    full_img = cv2.drawMatches(
        left_img,
        left_keypoints,
        right_img,
        right_keypoints,
        matches,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
    )
    return StereoImage(
        left=full_img[:, : full_img.shape[1] // 2],
        right=full_img[:, full_img.shape[1] // 2 :],
    )


def get_screen_size():
    # pylint:disable=import-outside-toplevel
    import tkinter

    root = tkinter.Tk()
    root.withdraw()
    width, height = root.winfo_screenwidth(), root.winfo_screenheight()
    return width, height


def _update_tracks(
    track_geometries: Dict[
        int, Tuple[o3d.geometry.PointCloud, o3d.geometry.LineSet, Color]
    ],
    visualizer: o3d.visualization.Visualizer,
    object_tracks: Dict[int, ObjectTrack],
    gt_trajectories: Dict[int, List[np.ndarray]],
    first_update: bool = False,
):
    LOGGER.debug("Displaying %d tracks", len(object_tracks))
    for ido, track in object_tracks.items():
        pt_cloud, bounding_box, path, gt_path, color = track_geometries.get(
            ido,
            (
                o3d.geometry.PointCloud(),
                o3d.geometry.LineSet(),
                o3d.geometry.LineSet(),
                o3d.geometry.LineSet(),
                COLORS[RNG.choice(len(COLORS))],
            ),
        )
        # only bright colors
        color[np.argmin(color)] = 0
        color[np.argmax(color)] = 1.0
        # draw path
        path_points = []
        gt_points = []
        gt_traj = gt_trajectories[ido]
        points = []
        white = np.array([1.0, 1.0, 1.0])
        lighter_color = color + 0.75 * white
        lighter_color = np.clip(lighter_color, 0, 1)
        track_size = 0
        for i, pose_world_obj in enumerate(reversed(list(track.poses.values()))):

            center = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
            for lm in track.landmarks.values():
                pt_world = from_homogeneous_pt(
                    pose_world_obj @ to_homogeneous_pt(lm.pt_3d)
                )
                center += pt_world
                if i == 0:
                    points.append(pt_world)
            if i == 0 and len(points) > 3:
                tmp_pt_cloud = o3d.geometry.PointCloud()
                tmp_pt_cloud.points = o3d.utility.Vector3dVector(points)
                bbox = tmp_pt_cloud.get_oriented_bounding_box()
                bbox.color = color
                bounding_box.points = bbox.get_box_points()
                bounding_box.lines = o3d.utility.Vector2iVector(
                    [
                        [0, 1],
                        [0, 2],
                        [0, 3],
                        [1, 6],
                        [1, 7],
                        [2, 5],
                        [2, 7],
                        [3, 5],
                        [3, 6],
                        [4, 5],
                        [4, 6],
                        [4, 7],
                    ]
                )
            if track.landmarks:
                center /= len(track.landmarks)
            path_points.append(center.reshape(3,).tolist())
            gt_points.append(gt_traj[i].reshape(3,).tolist())
            if i == 0:
                track_size = len(points)
            if ido == -1:
                break
        path_lines = [[i, i + 1] for i in range(len(path_points) - 1)]
        # draw current landmarks
        LOGGER.debug("Track has %d points", track_size)
        pt_cloud.points = o3d.utility.Vector3dVector(points)
        if len(path_lines) > 0:
            path.points = o3d.utility.Vector3dVector(path_points)
            path.lines = o3d.utility.Vector2iVector(path_lines)
            gt_path.points = o3d.utility.Vector3dVector(gt_points)
            gt_path.lines = o3d.utility.Vector2iVector(path_lines)
        if track.active:
            pt_cloud.paint_uniform_color(color)
            path.paint_uniform_color(color)
            gt_path.paint_uniform_color(lighter_color)
            bounding_box.paint_uniform_color(color)
        else:
            LOGGER.debug("Track is inactive")
            pt_cloud.paint_uniform_color([0.0, 0.0, 0.0])
            path.paint_uniform_color([0.0, 0.0, 0.0])
            gt_path.paint_uniform_color([0.0, 0.0, 0.0])
            bounding_box.paint_uniform_color([0.0, 0.0, 0.0])
        if track_geometries.get(ido) is None:
            visualizer.add_geometry(pt_cloud, reset_bounding_box=first_update)
            visualizer.add_geometry(path, reset_bounding_box=first_update)
            visualizer.add_geometry(gt_path, reset_bounding_box=first_update)
            visualizer.add_geometry(bounding_box, reset_bounding_box=first_update)
            track_geometries[ido] = (pt_cloud, bounding_box, path, gt_path, color)
        else:
            visualizer.update_geometry(pt_cloud)
            visualizer.update_geometry(path)
            visualizer.update_geometry(gt_path)
            visualizer.update_geometry(bounding_box)
    return track_geometries


def run(
    shared_data: queue.Queue,
    stop_flag: Event,
    next_step: Event,
    gt_trajectories: Dict[int, List[np.ndarray]],
):
    vis = o3d.visualization.Visualizer()
    width, height = get_screen_size()
    vis.create_window("MOT")
    view_control = vis.get_view_control()
    view_control.set_constant_z_far(100)
    view_control.set_constant_z_near(-100)
    opts = vis.get_render_option()
    opts.background_color = np.array([0.0, 0.0, 0.0,])
    cv2_window_name = "Stereo Image"
    cv2.namedWindow(cv2_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Stereo Image", (width // 2, height // 2))
    tracks: Dict[int, Tuple[o3d.geometry.PointCloud, Color]] = {}
    first_update = True
    while not stop_flag.is_set():
        try:
            new_data = shared_data.get_nowait()
        except queue.Empty:
            new_data = None
        if new_data is not None:
            LOGGER.debug("Got new data")
            tracks = _update_tracks(
                tracks,
                vis,
                new_data["object_tracks"],
                gt_trajectories,
                first_update=first_update,
            )
            stereo_image = new_data["stereo_image"]
            all_left_features = new_data["all_left_features"]
            all_right_features = new_data["all_right_features"]
            all_stereo_matches = new_data["all_stereo_matches"]
            # left_img = _draw_features(stereo_image.left, left_features)
            # right_img = _draw_features(stereo_image.right, right_features)
            # full_img = np.hstack([left_img, right_img])
            for left_features, right_features, stereo_matches in zip(
                all_left_features, all_right_features, all_stereo_matches
            ):
                stereo_image = _draw_matches(
                    stereo_image.left,
                    left_features,
                    stereo_image.right,
                    right_features,
                    stereo_matches,
                )
            full_img = np.hstack([stereo_image.left, stereo_image.right])
            cv2.imshow(cv2_window_name, full_img)
        keypress = cv2.waitKey(1)
        if keypress == ord("n"):
            next_step.set()

        vis.poll_events()
        vis.update_renderer()
        first_update = False
    LOGGER.debug("Finished viewer")
    vis.destroy_window("MOT")
