import logging
import queue
from threading import Event
from typing import Dict, List, Tuple

import cv2
import numpy as np
import open3d as o3d

from bamot.core.base_types import Feature, ObjectTrack, Match
from bamot.util.cv import from_homogeneous_pt, to_homogeneous_pt

LOGGER = logging.getLogger("UTIL:VIEWER")
RNG = np.random.default_rng()
Color = Tuple[float, float, float]
# Create some random colors
COLORS: List[Color] = list(
    map(lambda col: tuple(col / np.linalg.norm(col)), RNG.random((42, 3), dtype=float))
)


def _draw_features(img: np.ndarray, features: List[Feature]) -> np.ndarray:
    keypoints = [cv2.KeyPoint(x=f.u, y=f.v, _size=1) for f in features]
    return cv2.drawKeypoints(img, keypoints, None)


def _draw_matches(
    left_img: np.ndarray,
    left_features: List[Feature],
    right_img: np.ndarray,
    right_features: List[Feature],
    stereo_matches: List[Match],
) -> np.ndarray:
    left_keypoints = [cv2.KeyPoint(x=f.u, y=f.v, _size=1) for f in left_features]
    right_keypoints = [cv2.KeyPoint(x=f.u, y=f.v, _size=1) for f in right_features]
    # left is train, right is query?
    matches = [
        cv2.DMatch(_queryIdx=leftIdx, _trainIdx=rightIdx, _imgIdx=0, _distance=0)
        for leftIdx, rightIdx in stereo_matches[::20]
    ]
    return cv2.drawMatches(
        left_img, left_keypoints, right_img, right_keypoints, matches, None
    )


def get_screen_size():
    # pylint:disable=import-outside-toplevel
    import tkinter

    root = tkinter.Tk()
    root.withdraw()
    width, height = root.winfo_screenwidth(), root.winfo_screenheight()
    return width, height


def _update_tracks(
    track_geometries: Dict[int, Tuple[o3d.geometry.PointCloud, Color]],
    visualizer: o3d.visualization.Visualizer,
    object_tracks: Dict[int, ObjectTrack],
):
    LOGGER.debug("Displaying %d tracks", len(object_tracks))
    for ido, track in object_tracks.items():
        pose_world_obj = track.poses[max(track.poses.keys())]
        points = []
        for lm in track.landmarks.values():
            points.append(
                from_homogeneous_pt(
                    np.linalg.inv(pose_world_obj) @ to_homogeneous_pt(lm.pt_3d)
                )
            )
        LOGGER.debug("Track has %d points", len(points))
        pt_cloud, color = track_geometries.get(
            ido, (o3d.geometry.PointCloud(), COLORS[RNG.choice(len(COLORS))])
        )
        pt_cloud.points = o3d.utility.Vector3dVector(points)
        if track.active:
            LOGGER.debug("Track is active")
            pt_cloud.paint_uniform_color(color)
        else:
            LOGGER.debug("Track is inactive")
            pt_cloud.paint_uniform_color([0.0, 0.0, 0.0])
        if track_geometries.get(ido) is None:
            visualizer.add_geometry(pt_cloud)
            track_geometries[ido] = (pt_cloud, color)
        else:
            visualizer.update_geometry(pt_cloud)
    return track_geometries


def run(shared_data: queue.Queue, stop_flag: Event, next_step: Event):
    vis = o3d.visualization.Visualizer()
    width, height = get_screen_size()
    vis.create_window("MOT")
    cv2_window_name = "Stereo Image"
    cv2.namedWindow(cv2_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Stereo Image", (width // 2, height // 2))
    tracks: Dict[int, Tuple[o3d.geometry.PointCloud, Color]] = {}
    while not stop_flag.is_set():
        try:
            new_data = shared_data.get_nowait()
        except queue.Empty:
            new_data = None
        if new_data is not None:
            LOGGER.debug("Got new data")
            tracks = _update_tracks(tracks, vis, new_data["object_tracks"])
            stereo_image = new_data["stereo_image"]
            left_features = new_data["left_features"]
            right_features = new_data["right_features"]
            stereo_matches = new_data["stereo_matches"]
            # left_img = _draw_features(stereo_image.left, left_features)
            # right_img = _draw_features(stereo_image.right, right_features)
            # full_img = np.hstack([left_img, right_img])
            full_img = _draw_matches(
                stereo_image.left,
                left_features,
                stereo_image.right,
                right_features,
                stereo_matches,
            )
            cv2.imshow(cv2_window_name, full_img)
        keypress = cv2.waitKey(1)
        if keypress == ord("n"):
            next_step.set()

        vis.poll_events()
        vis.update_renderer()
    LOGGER.debug("Finished viewer")
    vis.destroy_window("MOT")
