import logging
import queue
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


class TrackGeometries(NamedTuple):
    pt_cloud: o3d.geometry.PointCloud
    trajectory: o3d.geometry.LineSet
    bbox: o3d.geometry.LineSet
    gt_trajectory: o3d.geometry.LineSet
    gt_bbox: o3d.geometry.LineSet
    color: Color


class EgoGeometries(NamedTuple):
    trajectory: o3d.geometry.LineSet
    curr_pose: o3d.geometry.TriangleMesh


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
    for ido, track in object_tracks.items():
        track_geometries = all_track_geometries.get(
            ido,
            TrackGeometries(
                pt_cloud=o3d.geometry.PointCloud(),
                trajectory=o3d.geometry.LineSet(),
                bbox=o3d.geometry.LineSet(),
                gt_trajectory=o3d.geometry.LineSet(),
                gt_bbox=o3d.geometry.LineSet(),
                color=_get_color(),
            ),
        )
        # draw path
        path_points = []
        gt_points = []
        track_data = label_data.get(ido)
        points = []
        white = np.array([1.0, 1.0, 1.0])
        color = track_geometries.color
        lighter_color = color + 0.25 * white
        lighter_color = np.clip(lighter_color, 0, 1)
        track_size = 0
        for i, (img_id, pose_world_obj) in enumerate(track.poses.items()):
            center = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
            for lm in track.landmarks.values():
                pt_world = from_homogeneous_pt(
                    pose_world_obj @ to_homogeneous_pt(lm.pt_3d)
                )
                center += pt_world
                if i == len(track.poses) - 1:
                    points.append(pt_world)
            if i == len(track.poses) - 1 and len(points) > 3:
                tmp_pt_cloud = o3d.geometry.PointCloud()
                tmp_pt_cloud.points = o3d.utility.Vector3dVector(points)
                bbox = tmp_pt_cloud.get_oriented_bounding_box()
                bbox.color = color
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
            if track.landmarks:
                center /= len(track.landmarks)
            path_points.append(center.reshape(3,).tolist())
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
        path_lines = [[i, i + 1] for i in range(len(path_points) - 1)]
        # draw current landmarks
        LOGGER.debug("Track has %d points", track_size)
        track_geometries.pt_cloud.points = o3d.utility.Vector3dVector(points)
        if len(path_lines) > 0:
            track_geometries.trajectory.points = o3d.utility.Vector3dVector(path_points)
            track_geometries.trajectory.lines = o3d.utility.Vector2iVector(path_lines)
            track_geometries.gt_trajectory.points = o3d.utility.Vector3dVector(
                gt_points
            )
            track_geometries.gt_trajectory.lines = o3d.utility.Vector2iVector(
                path_lines
            )
        if track.active:
            track_geometries.pt_cloud.paint_uniform_color(color)
            if show_trajs:
                track_geometries.trajectory.paint_uniform_color(color)
                if show_gt:
                    track_geometries.gt_trajectory.paint_uniform_color(lighter_color)
            track_geometries.bbox.paint_uniform_color(color)
            if show_gt:
                track_geometries.gt_bbox.paint_uniform_color(lighter_color)
        else:
            LOGGER.debug("Track is inactive")
            track_geometries.pt_cloud.paint_uniform_color([0.0, 0.0, 0.0])
            track_geometries.trajectory.paint_uniform_color([0.0, 0.0, 0.0])
            track_geometries.gt_trajectory.paint_uniform_color([0.0, 0.0, 0.0])
            track_geometries.bbox.paint_uniform_color([0.0, 0.0, 0.0])
            track_geometries.gt_bbox.paint_uniform_color([0.0, 0.0, 0.0])
        if all_track_geometries.get(ido) is None:
            visualizer.add_geometry(
                track_geometries.pt_cloud, reset_bounding_box=first_update
            )
            visualizer.add_geometry(
                track_geometries.trajectory, reset_bounding_box=first_update
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
            visualizer.update_geometry(track_geometries.trajectory)
            visualizer.update_geometry(track_geometries.gt_trajectory)
            visualizer.update_geometry(track_geometries.bbox)
            visualizer.update_geometry(track_geometries.gt_bbox)
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
                    [0.0000000, 0.0000000, 1.0000000, 10.0],
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
        if current_img_id > 0:
            ego_geometries.curr_pose.transform(
                np.linalg.inv(gt_poses[current_img_id - 1])
            )
        ego_geometries.curr_pose.transform(gt_poses[current_img_id])
        ego_geometries.curr_pose.paint_uniform_color(np.array([1.0, 1.0, 1.0]))
        visualizer.update_geometry(ego_geometries.trajectory)
        visualizer.update_geometry(ego_geometries.curr_pose)

    return all_track_geometries, ego_geometries


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
):
    if save_path:
        if not save_path.exists():
            save_path.mkdir(parents=True)
    vis = o3d.visualization.Visualizer()
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
        curr_pose=o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.25, cylinder_height=2, cone_radius=1.0, cone_height=1
        ),
    )
    vis.add_geometry(ego_geometries.trajectory)
    vis.add_geometry(ego_geometries.curr_pose)
    first_update = True
    counter = 0
    recording = False
    while not stop_flag.is_set():
        try:
            new_data = shared_data.get_nowait()
            shared_data.task_done()
        except queue.Empty:
            new_data = None
        if new_data is not None:
            LOGGER.debug("Got new data")
            (all_track_geometries, ego_geometries) = _update_geometries(
                all_track_geometries=all_track_geometries,
                ego_geometries=ego_geometries,
                visualizer=vis,
                object_tracks=new_data["object_tracks"],
                label_data=label_data,
                show_gt=show_gt,
                show_trajs=show_trajs,
                gt_poses=gt_poses,
                current_img_id=new_data["img_id"],
                first_update=first_update,
            )
            stereo_image = new_data["stereo_image"]
            all_left_features = new_data["all_left_features"]
            all_right_features = new_data["all_right_features"]
            all_stereo_matches = new_data["all_stereo_matches"]
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
    l, w, h = row.dim_3d
    rot_cam = row.rot_3d
    x_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]

    pts = []
    # w, h, l
    for x, y, z in zip(x_corners, y_corners, z_corners):
        pt_object = np.array([x, y, z]).reshape(3, 1)
        pt_cam = rot_cam @ pt_object
        pt_world = world_pos + T_world_cam[:3, :3] @ pt_cam
        pts.append(pt_world)

    pts = o3d.utility.Vector3dVector(pts)
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
