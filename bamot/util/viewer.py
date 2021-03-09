import logging
import queue
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Dict, List, NamedTuple, Optional, Set, Tuple, Union

import cv2
import numpy as np
import open3d as o3d
from bamot.config import CONFIG as config
from bamot.core.base_types import Feature, ObjectTrack, StereoImage, TrackId
from bamot.util.cv import (draw_contours, from_homogeneous,
                           get_corners_from_vector, to_homogeneous)
from bamot.util.kitti import DetectionData, DetectionDataRow
from bamot.util.misc import Color, get_color

LOGGER = logging.getLogger("UTIL:VIEWER")


class ViewerColors(NamedTuple):
    foreground: Tuple[float, float, float]
    background: Tuple[float, float, float]


VIEWER_COLORS: ViewerColors = None


@dataclass
class Colors:
    BLACK: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    WHITE: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class TrackGeometries:
    pt_cloud: o3d.geometry.PointCloud
    offline_trajectory: o3d.geometry.LineSet
    online_trajectory: o3d.geometry.LineSet
    bbox: o3d.geometry.LineSet
    color: Color


@dataclass
class EgoGeometries:
    trajectory: o3d.geometry.LineSet
    curr_pose: o3d.geometry.TriangleMesh
    curr_img: int


def _create_camera_lineset():
    lineset = o3d.geometry.LineSet()
    points = [[0, 0, 0], [2, 1, 2], [2, -1, 2], [-2, 1, 2], [-2, -1, 2]]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4]]
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    return lineset


def _enhance_image(
    stereo_img: StereoImage,
    all_left_features: List[Feature],
    all_right_features: List[Feature],
    tracks: Dict[TrackId, ObjectTrack],
    colors,
) -> StereoImage:
    for left_features, right_features in zip(all_left_features, all_right_features):
        left_keypoints = [cv2.KeyPoint(x=f.u, y=f.v, _size=1) for f in left_features]
        right_keypoints = [cv2.KeyPoint(x=f.u, y=f.v, _size=1) for f in right_features]
        cv2.drawKeypoints(stereo_img.left, left_keypoints, stereo_img.left)
        cv2.drawKeypoints(stereo_img.right, right_keypoints, stereo_img.right)
    for track_id, track in tracks.items():
        # opencv expects BGR non-normalized color as tuple
        clr = tuple((255 * np.flip(colors[track_id])).astype(int).tolist())
        shortend_track_id = (
            (str(track_id)[:3] + "..") if len(str(track_id)) > 5 else str(track_id)
        )
        if track.masks[0] is not None:
            stereo_img.left = draw_contours(track.masks[0], stereo_img.left, clr)
            y, x = map(min, np.where(track.masks[0] != 0))
            y_other, x_other = map(max, np.where(track.masks[0] != 0))
            stereo_img.left = cv2.putText(
                stereo_img.left,
                shortend_track_id,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                clr,
                3,
            )
            stereo_img.left = cv2.rectangle(
                stereo_img.left, (x, y), (x_other, y_other), clr, 1
            )
        if track.masks[1] is not None:
            stereo_img.right = draw_contours(track.masks[1], stereo_img.right, clr)
            y, x = map(min, np.where(track.masks[1] != 0))
            y_other, x_other = map(max, np.where(track.masks[1] != 0))
            stereo_img.right = cv2.putText(
                stereo_img.right,
                shortend_track_id,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                clr,
                3,
            )
            stereo_img.right = cv2.rectangle(
                stereo_img.right, (x, y), (x_other, y_other), clr, 1
            )

    return stereo_img


def get_screen_size():
    # pylint:disable=import-outside-toplevel
    import tkinter

    root = tkinter.Tk()
    root.withdraw()
    width, height = root.winfo_screenwidth(), root.winfo_screenheight()
    return width, height


def _update_gt_visualization(
    label_data,
    all_gt_track_geometries,
    visualizer,
    current_img_id,
    track_ids_match,
    all_track_geometries,
    gt_poses,
    show_gt,
    show_trajs,
):
    for track_id, track_data in label_data.items():
        if current_img_id not in track_data:
            if track_id in all_gt_track_geometries:
                geom = all_gt_track_geometries.pop(track_id)
                visualizer.remove_geometry(geom.bbox, reset_bounding_box=False)
                visualizer.remove_geometry(
                    geom.online_trajectory, reset_bounding_box=False
                )
        else:
            geometry = all_gt_track_geometries.get(
                track_id,
                TrackGeometries(
                    pt_cloud=None,
                    offline_trajectory=None,
                    online_trajectory=o3d.geometry.LineSet(),
                    bbox=o3d.geometry.LineSet(),
                    color=VIEWER_COLORS.foreground,
                ),
            )
            bbox_points, bbox_lines = _compute_bounding_box_from_kitti(
                track_data[current_img_id], gt_poses[current_img_id]
            )
            geometry.bbox.points = bbox_points
            geometry.bbox.lines = bbox_lines

            traj_points = np.asarray(geometry.online_trajectory.points).tolist()
            traj_points.append(track_data[current_img_id].world_pos)
            lines = [[i, i + 1] for i in range(len(traj_points) - 1)]

            geometry.online_trajectory.points = o3d.utility.Vector3dVector(traj_points)
            geometry.online_trajectory.lines = o3d.utility.Vector2iVector(lines)

            # paint black if GT is disabled
            if show_gt.val:
                color = (
                    all_track_geometries.get(track_id).color
                    if track_ids_match
                    and all_track_geometries.get(track_id) is not None
                    else VIEWER_COLORS.foreground
                )
                geometry.bbox.paint_uniform_color(color)
                if show_trajs:
                    geometry.online_trajectory.paint_uniform_color(color)
                else:
                    geometry.online_trajectory.paint_uniform_color(
                        VIEWER_COLORS.background
                    )
            else:
                geometry.bbox.paint_uniform_color(VIEWER_COLORS.background)
                geometry.online_trajectory.paint_uniform_color(VIEWER_COLORS.background)

            # update geometries in visualizer
            if all_gt_track_geometries.get(track_id) is None:
                visualizer.add_geometry(geometry.bbox, reset_bounding_box=False)
                all_gt_track_geometries[track_id] = geometry
            elif len(lines) == 1:
                visualizer.update_geometry(geometry.bbox)
                visualizer.add_geometry(
                    geometry.online_trajectory, reset_bounding_box=False
                )
            else:
                visualizer.update_geometry(geometry.bbox)
                visualizer.update_geometry(geometry.online_trajectory)


def _update_track_visualization(
    all_track_geometries,
    all_geometries,
    object_tracks,
    visualizer,
    show_online_trajs,
    show_offline_trajs,
    current_img_id,
    cached_colors,
    current_cam_pose,
):
    # display current track estimates
    inactive_tracks = []
    for ido in all_track_geometries:
        if ido not in object_tracks:
            inactive_tracks.append(ido)
    for ido, track in object_tracks.items():
        track_geometries = all_track_geometries.get(
            ido,
            TrackGeometries(
                pt_cloud=o3d.geometry.PointCloud(),
                offline_trajectory=o3d.geometry.LineSet(),
                online_trajectory=o3d.geometry.LineSet(),
                bbox=o3d.geometry.LineSet(),
                color=cached_colors.get(ido, get_color(only_bright=True)),
            ),
        )
        cached_colors[ido] = track_geometries.color
        # draw path
        path_points_offline = []
        path_points_online = []
        points = []
        color = track_geometries.color
        lighter_color = color + 0.25 * np.array(Colors.WHITE)
        darker_color = color - 0.25 * np.array(Colors.WHITE)
        lighter_color = np.clip(lighter_color, 0, 1)
        darker_color = np.clip(darker_color, 0, 1)
        track_size = 0
        dimensions = config.CAR_DIMS if track.cls == "car" else config.PED_DIMS
        for i, (img_id, pose_world_obj) in enumerate(track.poses.items()):
            # center = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
            if i == len(track.poses) - 1:
                for lm in track.landmarks.values():
                    pt_world = from_homogeneous(
                        pose_world_obj @ to_homogeneous(lm.pt_3d)
                    )
                    # Ecenter += pt_world
                    if i == len(track.poses) - 1 and np.isfinite(pt_world).all():
                        points.append(pt_world)
                if len(points) < 3:
                    continue
                try:
                    if track.rot_angle.get(current_img_id) is not None:
                        location = from_homogeneous(
                            np.linalg.inv(current_cam_pose)
                            @ to_homogeneous(track.locations[current_img_id])
                        )
                        # kitti locations are given as bottom of bounding box
                        # positive y direction is downward, hence add half of the dimensions
                        location[1] += dimensions[0] / 2
                        bbox, lines = _compute_bounding_box(
                            location,
                            track.rot_angle[current_img_id],
                            dimensions,
                            current_cam_pose,
                        )
                        track_geometries.bbox.points = bbox
                        track_geometries.bbox.lines = lines

                    else:
                        tmp_pt_cloud = o3d.geometry.PointCloud()
                        tmp_pt_cloud.points = o3d.utility.Vector3dVector(points)
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
            # if track.landmarks:
            #    center /= len(track.landmarks)
            # offline_point = center.reshape(3,).tolist()
            offline_point = track.pcl_centers.get(img_id)
            if offline_point is not None:
                pt_world = from_homogeneous(
                    pose_world_obj @ to_homogeneous(offline_point)
                )
                pt_world[2] -= dimensions[0] / 2
                path_points_offline.append(pt_world)
            online_point = track.locations.get(img_id).copy()
            if online_point is not None:
                online_point[2] -= dimensions[0] / 2
                path_points_online.append(online_point.reshape(3,).tolist())
            if i == len(track.poses) - 1:
                track_size = len(points)
        path_lines_offline = [[i, i + 1] for i in range(len(path_points_offline) - 1)]
        path_lines_online = [[i, i + 1] for i in range(len(path_points_online) - 1)]
        # draw current landmarks
        LOGGER.debug("Track has %d points", track_size)
        track_geometries.pt_cloud.points = o3d.utility.Vector3dVector(points)
        if path_lines_offline:
            track_geometries.offline_trajectory.points = o3d.utility.Vector3dVector(
                path_points_offline
            )
            track_geometries.offline_trajectory.lines = o3d.utility.Vector2iVector(
                path_lines_offline
            )
        if path_lines_online:
            track_geometries.online_trajectory.points = o3d.utility.Vector3dVector(
                path_points_online
            )
            track_geometries.online_trajectory.lines = o3d.utility.Vector2iVector(
                path_lines_online
            )
        if track.active:
            track_geometries.pt_cloud.paint_uniform_color(color)
            if show_offline_trajs.val:
                track_geometries.offline_trajectory.paint_uniform_color(color)
            else:
                track_geometries.offline_trajectory.paint_uniform_color(
                    VIEWER_COLORS.background
                )
            if show_online_trajs.val:
                track_geometries.online_trajectory.paint_uniform_color(darker_color)
            else:
                track_geometries.online_trajectory.paint_uniform_color(
                    VIEWER_COLORS.background
                )
            track_geometries.bbox.paint_uniform_color(color)
        else:
            LOGGER.debug("Track is inactive")
            track_geometries.pt_cloud.paint_uniform_color(VIEWER_COLORS.background)
            track_geometries.offline_trajectory.paint_uniform_color(
                VIEWER_COLORS.background
            )
            track_geometries.online_trajectory.paint_uniform_color(
                VIEWER_COLORS.background
            )
            track_geometries.bbox.paint_uniform_color(VIEWER_COLORS.background)
        if all_track_geometries.get(ido) is None:
            visualizer.add_geometry(track_geometries.pt_cloud, reset_bounding_box=False)
            visualizer.add_geometry(track_geometries.bbox, reset_bounding_box=False)
            all_geometries.add(track_geometries.pt_cloud)
            all_geometries.add(track_geometries.bbox)
            if (
                len(path_lines_online) >= 1
                and track_geometries.online_trajectory not in all_geometries
            ):
                visualizer.add_geometry(
                    track_geometries.online_trajectory, reset_bounding_box=False
                )
                all_geometries.add(track_geometries.online_trajectory)
            if (
                len(path_lines_offline) >= 1
                and track_geometries.offline_trajectory not in all_geometries
            ):
                visualizer.add_geometry(
                    track_geometries.offline_trajectory, reset_bounding_box=False
                )
                all_geometries.add(track_geometries.offline_trajectory)
            all_track_geometries[ido] = track_geometries
        else:
            visualizer.update_geometry(track_geometries.pt_cloud)
            if (
                len(path_lines_online) >= 1
                and track_geometries.online_trajectory not in all_geometries
            ):
                visualizer.add_geometry(
                    track_geometries.online_trajectory, reset_bounding_box=False
                )
                all_geometries.add(track_geometries.online_trajectory)
            else:
                visualizer.update_geometry(track_geometries.online_trajectory)
            if (
                len(path_lines_offline) >= 1
                and track_geometries.offline_trajectory not in all_geometries
            ):
                visualizer.add_geometry(
                    track_geometries.offline_trajectory, reset_bounding_box=False
                )
                all_geometries.add(track_geometries.offline_trajectory)
            else:
                visualizer.update_geometry(track_geometries.offline_trajectory)
            visualizer.update_geometry(track_geometries.bbox)

    for ido in inactive_tracks:
        track_geometry = all_track_geometries.get(ido)
        if track_geometry is not None:
            cached_colors[ido] = track_geometry.color
            visualizer.remove_geometry(
                track_geometry.pt_cloud, reset_bounding_box=False
            )
            visualizer.remove_geometry(track_geometry.bbox, reset_bounding_box=False)
            visualizer.remove_geometry(
                track_geometry.offline_trajectory, reset_bounding_box=False
            )
            visualizer.remove_geometry(
                track_geometry.online_trajectory, reset_bounding_box=False
            )

            all_geometries.remove(track_geometry.pt_cloud)
            all_geometries.remove(track_geometry.bbox)
            if track_geometry.offline_trajectory in all_geometries:
                all_geometries.remove(track_geometry.offline_trajectory)
            if track_geometry.online_trajectory in all_geometries:
                all_geometries.remove(track_geometry.online_trajectory)
            del all_track_geometries[ido]
            del cached_colors[ido]


def _update_ego_visualization(
    gt_poses, visualizer, ego_geometries, current_img_id, follow_ego=True
):
    ego_pts = []
    ego_path_lines = []
    for img_id, pose in enumerate(gt_poses):
        ego_pts.append(pose[:3, 3])
        if img_id == current_img_id:
            if follow_ego or current_img_id == 0:
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
        ego_geometries.trajectory.paint_uniform_color(VIEWER_COLORS.foreground)
        visualizer.update_geometry(ego_geometries.trajectory)
    if ego_geometries.curr_img < current_img_id:
        ego_geometries.curr_img = current_img_id
        if current_img_id > 0:
            ego_geometries.curr_pose.transform(
                np.linalg.inv(gt_poses[current_img_id - 1])
            )
        ego_geometries.curr_pose.transform(gt_poses[current_img_id])
        ego_geometries.curr_pose.paint_uniform_color(np.array(VIEWER_COLORS.foreground))
        visualizer.update_geometry(ego_geometries.curr_pose)


def _update_geometries(
    all_track_geometries: Dict[TrackId, TrackGeometries],
    all_gt_track_geometries: Dict[TrackId, TrackGeometries],
    all_geometries: Set[o3d.geometry.Geometry],
    ego_geometries: EgoGeometries,
    visualizer: o3d.visualization.Visualizer,
    object_tracks: Dict[TrackId, ObjectTrack],
    label_data: DetectionData,
    current_img_id: int,
    show_online_trajs: bool,
    show_offline_trajs: bool,
    show_gt: bool,
    cached_colors: Dict[TrackId, Color],
    gt_poses: List[np.ndarray],
    track_ids_match: bool = False,
    follow_ego: bool = True,
) -> Tuple[Dict[int, TrackGeometries], EgoGeometries]:
    LOGGER.debug("Displaying %d tracks", len(object_tracks))
    if label_data is not None:
        # display all GT tracks present in current img
        _update_gt_visualization(
            label_data=label_data,
            all_gt_track_geometries=all_gt_track_geometries,
            visualizer=visualizer,
            all_track_geometries=all_track_geometries,
            show_gt=show_gt,
            show_trajs=(show_online_trajs.val or show_offline_trajs.val),
            gt_poses=gt_poses,
            track_ids_match=track_ids_match,
            current_img_id=current_img_id,
        )

    # display all track estimates
    _update_track_visualization(
        all_track_geometries=all_track_geometries,
        all_geometries=all_geometries,
        object_tracks=object_tracks,
        visualizer=visualizer,
        show_online_trajs=show_online_trajs,
        show_offline_trajs=show_offline_trajs,
        current_img_id=current_img_id,
        cached_colors=cached_colors,
        current_cam_pose=gt_poses[current_img_id],
    )

    # display ego camera and trajectory
    _update_ego_visualization(
        gt_poses=gt_poses,
        visualizer=visualizer,
        ego_geometries=ego_geometries,
        current_img_id=current_img_id,
        follow_ego=follow_ego,
    )


@dataclass
class Boolean:
    val: bool


def _toggle(boolean: Union[List[Boolean], Boolean], geometry_has_changed):
    if isinstance(boolean, list):
        # set both to false if both are true, else false
        both_true = all([b.val for b in boolean])
        for b in boolean:
            b.val = not both_true
    else:
        boolean.val = not boolean.val
    geometry_has_changed.val = True
    return True


def run(
    shared_data: queue.Queue,
    stop_flag: Event,
    next_step: Event,
    label_data: DetectionData,
    gt_poses: Optional[List[np.ndarray]] = None,
    save_path: Optional[Path] = None,
    trajs: str = "both",
    show_gt: bool = True,
    recording: bool = False,
    track_ids_match: bool = False,
    dark_background: bool = False,
):
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        save_path_3d = save_path / "3d"
        save_path_2d = save_path / "2d"
        save_path_3d.mkdir(exist_ok=True)
        save_path_2d.mkdir(exist_ok=True)
    global VIEWER_COLORS
    if dark_background:
        VIEWER_COLORS = ViewerColors(foreground=Colors.WHITE, background=Colors.BLACK)
    else:
        VIEWER_COLORS = ViewerColors(foreground=Colors.BLACK, background=Colors.WHITE)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    show_gt = Boolean(show_gt)
    show_offline_trajs = Boolean(trajs in ["both", "offline"])
    show_online_trajs = Boolean(trajs in ["both", "online"])
    geometry_has_changed = Boolean(False)
    B = 66
    N = 78  # online
    F = 70  # offline
    G = 71  # GT
    vis.register_key_callback(
        N, lambda vis: _toggle(show_online_trajs, geometry_has_changed)
    )
    vis.register_key_callback(
        F, lambda vis: _toggle(show_offline_trajs, geometry_has_changed)
    )
    vis.register_key_callback(
        B,
        lambda vis: _toggle(
            [show_offline_trajs, show_online_trajs], geometry_has_changed
        ),
    )
    vis.register_key_callback(G, lambda vis: _toggle(show_gt, geometry_has_changed))
    width, height = get_screen_size()
    vis.create_window("MOT", top=0, left=1440)
    view_control = vis.get_view_control()
    view_control.set_constant_z_far(150)
    view_control.set_constant_z_near(-10)
    opts = vis.get_render_option()
    opts.background_color = np.array(VIEWER_COLORS.background)
    opts.point_size = 2.0
    opts.line_width = 50.0
    cv2_window_name = "Stereo Image"
    cv2.namedWindow(cv2_window_name, cv2.WINDOW_NORMAL)
    all_track_geometries: Dict[int, TrackGeometries] = {}
    all_gt_track_geometries: Dict[int, TrackGeometries] = {}
    ego_geometries = EgoGeometries(
        trajectory=o3d.geometry.LineSet(),
        curr_pose=_create_camera_lineset(),
        curr_img=-1,
    )
    vis.add_geometry(ego_geometries.trajectory, reset_bounding_box=False)
    vis.add_geometry(ego_geometries.curr_pose)
    first_update = True
    counter = 0
    object_tracks = {}
    cached_colors = {}
    current_img_id = 0
    all_geometries = set()
    while not stop_flag.is_set() or not shared_data.empty():
        try:
            new_data = shared_data.get_nowait()
            shared_data.task_done()
        except queue.Empty:
            new_data = {}
        if new_data or geometry_has_changed.val:
            geometry_has_changed.val = False
            if new_data:
                object_tracks = new_data.get("object_tracks", object_tracks)
                current_img_id = new_data.get("img_id", current_img_id)
            LOGGER.debug("Got new data")
            _update_geometries(
                all_track_geometries=all_track_geometries,
                all_gt_track_geometries=all_gt_track_geometries,
                all_geometries=all_geometries,
                ego_geometries=ego_geometries,
                visualizer=vis,
                object_tracks=object_tracks,
                label_data=label_data,
                show_gt=show_gt,
                show_offline_trajs=show_offline_trajs,
                show_online_trajs=show_online_trajs,
                gt_poses=gt_poses,
                current_img_id=current_img_id,
                track_ids_match=track_ids_match,
                cached_colors=cached_colors,
                follow_ego=recording,
            )
            if new_data:
                stereo_image = new_data["stereo_image"]
                all_left_features = new_data.get("all_left_features", [])
                all_right_features = new_data.get("all_right_features", [])
                stereo_image = _enhance_image(
                    stereo_image,
                    all_left_features,
                    all_right_features,
                    tracks=object_tracks,
                    colors=cached_colors,
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
                        (save_path_3d / f"{counter:06}.png").as_posix(), False
                    )
                    cv2.imwrite(
                        (save_path_2d / f"{counter:06}.png").as_posix(),
                        stereo_image.left,
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


def _compute_bounding_box_from_kitti(row: DetectionDataRow, T_world_cam: np.ndarray):
    return _compute_bounding_box(
        location=row.cam_pos,
        rot_angle=row.rot_angle,
        dimensions=row.dim_3d,
        T_world_cam=T_world_cam,
    )


def _compute_bounding_box(location, rot_angle, dimensions, T_world_cam):
    vec = np.array([*location, rot_angle, *dimensions]).reshape(7, 1)
    corners_cam = get_corners_from_vector(vec)
    corners_world = from_homogeneous(T_world_cam @ to_homogeneous(corners_cam))

    return _get_points_and_lines_from_corners(corners_world)


def _get_points_and_lines_from_corners(corners: np.ndarray):
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


def visualize_pointcloud_and_obb(
    pointcloud: np.ndarray, corners: Union[List[np.ndarray], np.ndarray]
):
    vis = o3d.visualization.Visualizer()
    vis.create_window("MOT", top=0, left=1440)
    opts = vis.get_render_option()
    opts.background_color = np.array(Colors.BLACK)
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(pointcloud.T)
    vis.add_geometry(pcl)
    if isinstance(corners, list):
        for c in corners:
            obb = o3d.geometry.LineSet()
            pts, lines = _get_points_and_lines_from_corners(c)
            obb.lines = lines
            obb.points = pts
            obb.paint_uniform_color(Colors.WHITE)
            vis.add_geometry(obb)
    else:
        obb = o3d.geometry.LineSet()
        pts, lines = _get_points_and_lines_from_corners(corners)
        obb.lines = lines
        obb.points = pts
        obb.paint_uniform_color(Colors.WHITE)
        vis.add_geometry(obb)
    vis.run()
    vis.destroy_window()
