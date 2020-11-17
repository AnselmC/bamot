import logging
import queue
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Dict, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
from bamot.core.base_types import Feature, Match, ObjectTrack, StereoImage
from bamot.util.cv import from_homogeneous_pt, to_homogeneous_pt
from bamot.util.kitti import LabelData, LabelDataRow

LOGGER = logging.getLogger("UTIL:VIEWER")
RNG = np.random.default_rng()
Color = np.ndarray
# Create some random colors
COLORS: List[Color] = RNG.random((42, 3))
BLACK = [0.0, 0.0, 0.0]


class TrackGeometries(NamedTuple):
    pt_cloud: o3d.geometry.PointCloud
    offline_trajectory: o3d.geometry.LineSet
    online_trajectory: o3d.geometry.LineSet
    bbox: o3d.geometry.LineSet
    gt_trajectory: o3d.geometry.LineSet
    gt_bbox: o3d.geometry.LineSet
    color: Color


@dataclass
class EgoGeometries:
    trajectory: o3d.geometry.LineSet
    curr_pose: o3d.geometry.TriangleMesh
    curr_img: int


def _create_camera_lineset():
    lineset = o3d.geometry.LineSet()
    points = [[0, 0, 2], [2, 1, 3], [2, -1, 3], [-2, 1, 3], [-2, -1, 3]]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4]]
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    return lineset


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


def _get_color():
    color = COLORS[RNG.choice(len(COLORS))]
    # only bright colors
    color[np.argmin(color)] = 0
    color[np.argmax(color)] = 1.0
    return color


def _update_geometries(
    all_track_geometries: Dict[int, TrackGeometries],
    ego_geometries: EgoGeometries,
    visualizer: o3d.visualization.Visualizer,
    object_tracks: Dict[int, ObjectTrack],
    label_data: LabelData,
    current_img_id: int,
    show_trajs: bool,
    show_gt: bool,
    gt_poses: List[np.ndarray],
    first_update: bool = False,
) -> Tuple[Dict[int, TrackGeometries], EgoGeometries]:
    LOGGER.debug("Displaying %d tracks", len(object_tracks))
    inactive_tracks = []
    for ido, track in object_tracks.items():
        if not track.active:
            inactive_tracks.append(ido)
            continue
        track_geometries = all_track_geometries.get(
            ido,
            TrackGeometries(
                pt_cloud=o3d.geometry.PointCloud(),
                offline_trajectory=o3d.geometry.LineSet(),
                online_trajectory=o3d.geometry.LineSet(),
                bbox=o3d.geometry.LineSet(),
                gt_trajectory=o3d.geometry.LineSet(),
                gt_bbox=o3d.geometry.LineSet(),
                color=_get_color(),
            ),
        )
        # draw path
        path_points_offline = []
        path_points_online = []
        gt_points = []
        track_data = label_data.get(ido)
        points = []
        white = np.array([1.0, 1.0, 1.0])
        color = track_geometries.color
        lighter_color = color + 0.25 * white
        darker_color = color - 0.25 * white
        lighter_color = np.clip(lighter_color, 0, 1)
        darker_color = np.clip(darker_color, 0, 1)
        track_size = 0
        for i, (img_id, pose_world_obj) in enumerate(track.poses.items()):
            center = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
            for lm in track.landmarks.values():
                pt_world = from_homogeneous_pt(
                    pose_world_obj @ to_homogeneous_pt(lm.pt_3d)
                )
                center += pt_world
                if i == len(track.poses) - 1 and np.isfinite(pt_world).all():
                    points.append(pt_world)
            if i == len(track.poses) - 1 and len(points) > 3:
                tmp_pt_cloud = o3d.geometry.PointCloud()
                tmp_pt_cloud.points = o3d.utility.Vector3dVector(points)
                try:
                    bbox = tmp_pt_cloud.get_oriented_bounding_box()
                    track_geometries.bbox.points = bbox.get_box_points()
                    track_geometries.bbox.lines = o3d.utility.Vector2iVector(
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
                except RuntimeError:
                    # happens when points are too close to each other
                    pass
            if track.landmarks:
                center /= len(track.landmarks)
            offline_point = center.reshape(3,).tolist()
            path_points_offline.append(offline_point)
            path_points_online.append(
                from_homogeneous_pt(
                    track.locations.get(img_id, to_homogeneous_pt(center))
                )
            )
            if track_data:
                if track_data.get(img_id):
                    gt_point = track_data[img_id].world_pos
                    gt_points.append(gt_point)
                    # TODO: don't update every step
                    gt_bbox_points, gt_bbox_lines = _compute_bounding_box_from_kitti(
                        track_data[img_id], gt_poses[img_id]
                    )
                    track_geometries.gt_bbox.points = gt_bbox_points
                    track_geometries.gt_bbox.lines = gt_bbox_lines
            if i == len(track.poses) - 1:
                track_size = len(points)
            if ido == -1:
                break
        path_lines = [[i, i + 1] for i in range(len(path_points_offline) - 1)]
        # draw current landmarks
        LOGGER.debug("Track has %d points", track_size)
        track_geometries.pt_cloud.points = o3d.utility.Vector3dVector(points)
        if len(path_lines) > 0:
            track_geometries.offline_trajectory.points = o3d.utility.Vector3dVector(
                path_points_offline
            )
            track_geometries.offline_trajectory.lines = o3d.utility.Vector2iVector(
                path_lines
            )
            track_geometries.online_trajectory.points = o3d.utility.Vector3dVector(
                path_points_online
            )
            track_geometries.online_trajectory.lines = o3d.utility.Vector2iVector(
                path_lines
            )
            track_geometries.gt_trajectory.points = o3d.utility.Vector3dVector(
                gt_points
            )
            track_geometries.gt_trajectory.lines = o3d.utility.Vector2iVector(
                path_lines
            )
        if track.active:
            track_geometries.pt_cloud.paint_uniform_color(color)
            if show_trajs.val:
                track_geometries.offline_trajectory.paint_uniform_color(color)
                track_geometries.online_trajectory.paint_uniform_color(darker_color)
                if show_gt.val:
                    track_geometries.gt_trajectory.paint_uniform_color(lighter_color)
                else:
                    track_geometries.gt_trajectory.paint_uniform_color(BLACK)
            else:
                track_geometries.offline_trajectory.paint_uniform_color(BLACK)
                track_geometries.online_trajectory.paint_uniform_color(BLACK)
                track_geometries.gt_trajectory.paint_uniform_color(BLACK)
            track_geometries.bbox.paint_uniform_color(color)
            if show_gt.val:
                track_geometries.gt_bbox.paint_uniform_color(lighter_color)
            else:
                track_geometries.gt_bbox.paint_uniform_color(BLACK)
        else:
            LOGGER.debug("Track is inactive")
            track_geometries.pt_cloud.paint_uniform_color([0.0, 0.0, 0.0])
            track_geometries.offline_trajectory.paint_uniform_color([0.0, 0.0, 0.0])
            track_geometries.online_trajectory.paint_uniform_color([0.0, 0.0, 0.0])
            track_geometries.gt_trajectory.paint_uniform_color([0.0, 0.0, 0.0])
            track_geometries.bbox.paint_uniform_color([0.0, 0.0, 0.0])
            track_geometries.gt_bbox.paint_uniform_color([0.0, 0.0, 0.0])
        if all_track_geometries.get(ido) is None:
            visualizer.add_geometry(
                track_geometries.pt_cloud, reset_bounding_box=first_update
            )
            visualizer.add_geometry(
                track_geometries.offline_trajectory, reset_bounding_box=first_update
            )
            visualizer.add_geometry(
                track_geometries.online_trajectory, reset_bounding_box=first_update
            )
            visualizer.add_geometry(
                track_geometries.gt_trajectory, reset_bounding_box=first_update
            )
            visualizer.add_geometry(
                track_geometries.bbox, reset_bounding_box=first_update
            )
            visualizer.add_geometry(
                track_geometries.gt_bbox, reset_bounding_box=first_update
            )
            all_track_geometries[ido] = track_geometries
        else:
            visualizer.update_geometry(track_geometries.pt_cloud)
            visualizer.update_geometry(track_geometries.offline_trajectory)
            visualizer.update_geometry(track_geometries.online_trajectory)
            visualizer.update_geometry(track_geometries.gt_trajectory)
            visualizer.update_geometry(track_geometries.bbox)
            visualizer.update_geometry(track_geometries.gt_bbox)

    for ido in inactive_tracks:
        track_geometry = all_track_geometries.get(ido)
        if track_geometry is not None:
            visualizer.remove_geometry(track_geometry.pt_cloud)
            visualizer.remove_geometry(track_geometry.bbox)
            visualizer.remove_geometry(track_geometry.offline_trajectory)
            visualizer.remove_geometry(track_geometry.online_trajectory)
            visualizer.remove_geometry(track_geometry.gt_bbox)
            visualizer.remove_geometry(track_geometry.gt_trajectory)
            del all_track_geometries[ido]
    ego_pts = []
    ego_path_lines = []
    for img_id, pose in enumerate(gt_poses):
        ego_pts.append(pose[:3, 3])
        if img_id > current_img_id:
            ctr = visualizer.get_view_control()
            cam_parameters = ctr.convert_to_pinhole_camera_parameters()
            init_traf = np.array(
                [
                    [1.0000000, 0.0000000, 0.0000000, 0.0],
                    [0.0000000, 1.0000000, 0.0000000, 6.0],
                    [0.0000000, 0.0000000, 1.0000000, 15.0],
                    [0, 0, 0, 1],
                ]
            )
            cam_parameters.extrinsic = init_traf @ np.linalg.inv(pose)
            ctr.convert_from_pinhole_camera_parameters(cam_parameters)
            break
        ego_path_lines.append([img_id, img_id + 1])
    if len(ego_path_lines) > 0:
        ego_geometries.trajectory.points = o3d.utility.Vector3dVector(ego_pts)
        ego_geometries.trajectory.lines = o3d.utility.Vector2iVector(ego_path_lines)
        ego_geometries.trajectory.paint_uniform_color(np.array([1.0, 1.0, 1.0]))
        if ego_geometries.curr_img < current_img_id:
            ego_geometries.curr_img = current_img_id
            if current_img_id > 0:
                ego_geometries.curr_pose.transform(
                    np.linalg.inv(gt_poses[current_img_id - 1])
                )
            ego_geometries.curr_pose.transform(gt_poses[current_img_id])
            ego_geometries.curr_pose.paint_uniform_color(np.array([1.0, 1.0, 1.0]))
            visualizer.update_geometry(ego_geometries.trajectory)
            visualizer.update_geometry(ego_geometries.curr_pose)

    return all_track_geometries, ego_geometries


@dataclass
class Boolean:
    val: bool


def _toggle(boolean: Boolean, geometry_has_changed):
    boolean.val = not boolean.val
    geometry_has_changed.val = True
    return True


def run(
    shared_data: queue.Queue,
    stop_flag: Event,
    next_step: Event,
    label_data: LabelData,
    gt_poses: Optional[List[np.ndarray]] = None,
    save_path: Optional[Path] = None,
    cam_coordinates: bool = False,
    show_trajs: bool = True,
    show_gt: bool = True,
    recording: bool = False,
):
    if save_path:
        if not save_path.exists():
            save_path.mkdir(parents=True)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    show_gt = Boolean(show_gt)
    show_trajs = Boolean(show_trajs)
    geometry_has_changed = Boolean(False)
    vis.register_key_callback(84, lambda vis: _toggle(show_trajs, geometry_has_changed))
    vis.register_key_callback(71, lambda vis: _toggle(show_gt, geometry_has_changed))
    width, height = get_screen_size()
    vis.create_window("MOT", top=0, left=1440)
    view_control = vis.get_view_control()
    view_control.set_constant_z_far(150)
    view_control.set_constant_z_near(-10)
    opts = vis.get_render_option()
    opts.background_color = np.array([0.0, 0.0, 0.0,])
    cv2_window_name = "Stereo Image"
    cv2.namedWindow(cv2_window_name, cv2.WINDOW_NORMAL)
    all_track_geometries: Dict[int, TrackGeometries] = {}
    ego_geometries = EgoGeometries(
        trajectory=o3d.geometry.LineSet(),
        curr_pose=_create_camera_lineset(),
        curr_img=-1,
    )
    vis.add_geometry(ego_geometries.trajectory)
    vis.add_geometry(ego_geometries.curr_pose)
    first_update = True
    counter = 0
    object_tracks = {}
    current_img_id = 0
    while not stop_flag.is_set() or not shared_data.empty():
        try:
            new_data = shared_data.get_nowait()
            shared_data.task_done()
        except queue.Empty:
            new_data = None
        if new_data is not None or geometry_has_changed.val:
            geometry_has_changed.val = False
            if new_data is not None:
                object_tracks = new_data.get("object_tracks", object_tracks)
                current_img_id = new_data.get("img_id", current_img_id)
            LOGGER.debug("Got new data")
            (all_track_geometries, ego_geometries) = _update_geometries(
                all_track_geometries=all_track_geometries,
                ego_geometries=ego_geometries,
                visualizer=vis,
                object_tracks=object_tracks,
                label_data=label_data,
                show_gt=show_gt,
                show_trajs=show_trajs,
                gt_poses=gt_poses,
                current_img_id=current_img_id,
                first_update=first_update,
            )
            if new_data is not None:
                stereo_image = new_data["stereo_image"]
                all_left_features = new_data.get("all_left_features", [])
                all_right_features = new_data.get("all_right_features", [])
                all_stereo_matches = new_data.get("all_stereo_matches", [])
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
                if first_update:
                    img_size = stereo_image.left.shape
                    img_ratio = img_size[1] / img_size[0]
                    img_width = int(0.75 * width)
                    img_height = int(img_width / img_ratio)
                    cv2.resizeWindow("Stereo Image", (img_width, img_height))

                full_img = np.hstack([stereo_image.left, stereo_image.right])
                cv2.imshow(cv2_window_name, full_img)
                if save_path is not None and recording:
                    LOGGER.debug("Saving image: %s", counter)
                    vis.capture_screen_image(
                        (save_path / f"{counter:06}.png").as_posix(), False
                    )
                    counter += 1

        keypress = cv2.waitKey(1)
        if keypress == ord("n"):
            next_step.set()
        if keypress == ord("s"):
            recording = not recording
            LOGGER.debug("Recording: %s", recording)
        vis.poll_events()
        vis.update_renderer()
        first_update = False
    vis.destroy_window()
    cv2.destroyAllWindows()
    LOGGER.debug("Finished viewer")


def _compute_bounding_box_from_kitti(row: LabelDataRow, T_world_cam):
    world_pos = row.world_pos
    # whl
    h, w, l = row.dim_3d
    rot_cam = row.rot_3d
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners = np.array([x_corners, y_corners, z_corners])

    corners = (T_world_cam[:3, :3] @ rot_cam @ corners) + world_pos

    pts = o3d.utility.Vector3dVector(corners.T)
    lines = o3d.utility.Vector2iVector(
        [
            [0, 1],
            [0, 3],
            [0, 4],
            [1, 2],
            [1, 5],
            [2, 3],
            [2, 6],
            [3, 7],
            [4, 5],
            [4, 7],
            [6, 7],
            [5, 6],
        ]
    )
    return pts, lines
