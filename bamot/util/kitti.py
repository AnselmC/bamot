import glob
import logging
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as rletools
from bamot.core.base_types import (CameraParameters, ImageId, ObjectDetection,
                                   StereoCamera, StereoImage,
                                   StereoObjectDetection, TrackId)
from bamot.util.cv import from_homogeneous_pt, to_homogeneous_pt
from g2o import AngleAxis

LOGGER = logging.getLogger("Util:Kitti")


class LabelDataRow(NamedTuple):
    world_pos: np.ndarray
    cam_pos: np.ndarray
    occ_lvl: int
    trunc_lvl: int
    bbox2d: Tuple[float, float, float, float]
    object_class: str
    dim_3d: Tuple[float, float, float]
    rot_3d: np.ndarray


LabelData = Dict[TrackId, Dict[ImageId, LabelDataRow]]


def get_detection_stream(
    obj_detections_path: Path,
    offset: int,
    label_data: Optional[LabelData] = None,
    object_ids: Optional[List[int]] = None,
) -> Iterable[List[StereoObjectDetection]]:
    detection_files = sorted(glob.glob(obj_detections_path.as_posix() + "/*.pkl"))
    if not detection_files:
        raise ValueError(f"No detection files found at {obj_detections_path}")
    LOGGER.debug("Found %d detection files", len(detection_files))
    for i, f in enumerate(detection_files[offset:]):
        with open(f, "rb") as fp:
            detections = pickle.load(fp)
        if object_ids:
            detections = [d for d in detections if d.left.track_id in object_ids]

        if label_data:  # only available for GT tracks
            for d in detections:
                try:
                    row_data = label_data[d.left.track_id][i + offset]
                    d.fully_visible = row_data.occ_lvl <= 1 and row_data.trunc_lvl <= 1
                except KeyError as exc:
                    # TODO: why is this happening?
                    pass
        yield detections
    LOGGER.debug("Finished yielding object detections")


def get_image_shape(kitti_path: str, scene: str) -> Tuple[int, int]:
    left_img_path = Path(kitti_path) / "image_02" / scene
    left_imgs = sorted(glob.glob(left_img_path.as_posix() + "/*.png"))
    img_shape = cv2.imread(left_imgs[0], cv2.IMREAD_COLOR).astype(np.uint8).shape
    return img_shape


def get_image_stream(
    kitti_path: Path, scene: str, offset: int = 0, with_file_names: bool = False,
) -> Iterable[StereoImage]:
    class Stream:
        def __init__(self, generator, length):
            self._generator = generator
            self._length = length

        def __len__(self):
            return self._length

        def __iter__(self):
            return self._generator

    left_img_path = kitti_path / "image_02" / scene
    right_img_path = kitti_path / "image_03" / scene
    left_imgs = sorted(glob.glob(left_img_path.as_posix() + "/*.png"))[offset:]
    right_imgs = sorted(glob.glob(right_img_path.as_posix() + "/*.png"))[offset:]

    def _generator(with_file_names: bool):
        for left, right in zip(left_imgs, right_imgs):
            left_img = cv2.imread(left, cv2.IMREAD_COLOR).astype(np.uint8)
            right_img = cv2.imread(right, cv2.IMREAD_COLOR).astype(np.uint8)
            if with_file_names:
                yield StereoImage(left_img, right_img), (left, right)
            else:
                yield StereoImage(left_img, right_img)

    return Stream(_generator(with_file_names), len(left_imgs))


def get_label_data_from_kitti(
    kitti_path: Path,
    scene: str,
    poses: List[np.ndarray],
    offset: int = 0,
    indexed_by_image_id: bool = False,
) -> LabelData:
    detection_file = _get_detection_file(kitti_path, scene)
    if not detection_file.exists():
        raise FileNotFoundError(
            f"Detection file {detection_file.as_posix()} doesn't exist"
        )
    label_data: LabelData = {}
    with open(detection_file.as_posix(), "r") as fp:
        for line in fp:
            cols = line.split(" ")
            frame = int(cols[0])
            if frame < offset:
                continue
            track_id = int(cols[1])
            if track_id == -1:
                continue
            object_class = str(cols[2])
            if object_class not in ["Car", "Pedestrian"]:
                continue
            truncation_level = int(cols[3])
            occlusion_level = int(cols[4])
            # cols[5] is observation angle of object
            bbox = list(map(float, cols[6:10]))  # in left image coordinates
            dim_3d = list(map(float, cols[10:13]))  # in camera coordinates
            location_cam2 = np.array(list(map(float, cols[13:16]))).reshape(
                (3, 1)
            )  # in camera coordinates
            rot_angle = float(cols[16])  # in camera coordinates
            rot_3d = AngleAxis(rot_angle, np.array([0, 1, 0])).rotation_matrix()
            T_w_cam2 = poses[frame]
            location_world = from_homogeneous_pt(
                T_w_cam2 @ to_homogeneous_pt(location_cam2)
            )
            if indexed_by_image_id:
                if not label_data.get(frame):
                    label_data[frame] = {}
            else:
                if not label_data.get(track_id):
                    label_data[track_id] = {}

            label_row = LabelDataRow(
                world_pos=location_world.tolist(),
                cam_pos=location_cam2.tolist(),
                occ_lvl=occlusion_level,
                trunc_lvl=truncation_level,
                object_class=object_class,
                bbox2d=bbox,
                dim_3d=dim_3d,
                rot_3d=rot_3d,
            )
            if indexed_by_image_id:
                label_data[frame][track_id] = label_row
            else:
                label_data[track_id][frame] = label_row

    LOGGER.debug("Extracted GT trajectories for %d objects", len(label_data))
    return label_data


def get_gt_poses_from_kitti(kitti_path: Path, scene: str) -> List[np.ndarray]:
    """Adapted from Sergio Agostinho, returns GT poses of left cam"""
    oxts_file = _get_oxts_file(kitti_path, scene)
    poses = []
    if not oxts_file.exists():
        raise FileNotFoundError(oxts_file.as_posix())

    cols = (
        "lat",
        "lon",
        "alt",
        "roll",
        "pitch",
        "yaw",
        "vn",
        "ve",
        "vf",
        "vl",
        "vu",
        "ax",
        "ay",
        "az",
        "af",
        "al",
        "au",
        "wx",
        "wy",
        "wz",
        "wf",
        "wl",
        "wu",
        "posacc",
        "velacc",
        "navstat",
        "numsats",
        "posmode",
        "velmode",
        "orimode",
    )
    df = pd.read_csv(oxts_file.as_posix(), sep=" ", names=cols, index_col=False)
    Tr_cam_imu = get_transformation_cam_to_imu(kitti_path, scene)
    poses = [
        T_w_imu @ np.linalg.inv(Tr_cam_imu)
        for T_w_imu in _oxts_to_poses(
            *df[["lat", "lon", "alt", "roll", "pitch", "yaw"]].values.T
        )
    ]
    return poses


def get_transformation_cam_to_imu(kitti_path: Path, scene) -> np.ndarray:
    calib_file = _get_calib_file(kitti_path, scene)
    with open(calib_file.as_posix(), "r") as fp:
        for line in fp:
            cols = line.split(" ")
            name = cols[0]
            if name == "Tr_imu_velo":
                Tr_imu_velo = np.array(list(map(float, cols[1:-2]))).reshape(3, 4)
                Tr_imu_velo = np.vstack([Tr_imu_velo, np.array([[0, 0, 0, 1]])])
            elif name == "Tr_velo_cam":
                Tr_velo_cam = np.array(list(map(float, cols[1:-2]))).reshape(3, 4)
                Tr_velo_cam = np.vstack([Tr_velo_cam, np.array([[0, 0, 0, 1]])])

    Tr_imu_cam = Tr_imu_velo @ Tr_velo_cam
    return Tr_imu_cam


def get_cameras_from_kitti(kitti_path: Path) -> Tuple[StereoCamera, np.ndarray]:
    calib_file = _get_calib_cam_to_cam_file(kitti_path)
    with open(calib_file.as_posix(), "r") as fp:
        for line in fp:
            cols = line.split(" ")
            name = cols[0]
            # if "R_02" in name:
            #    R02 = np.array(list(map(float, cols[1:]))).reshape(3, 3)
            # elif "T_02" in name:
            #    t02 = np.array(list(map(float, cols[1:]))).reshape(3)
            # elif "R_03" in name:
            #    R03 = np.array(list(map(float, cols[1:]))).reshape(3, 3)
            # elif "T_03" in name:
            #    t03 = np.array(list(map(float, cols[1:]))).reshape(3)
            if "P_rect_02" in name:
                P_rect_left = np.array(list(map(float, cols[1:]))).reshape(3, 4)
            elif "P_rect_03" in name:
                P_rect_right = np.array(list(map(float, cols[1:]))).reshape(3, 4)
            elif "R_rect_02" in name:
                R_rect_02 = np.array(list(map(float, cols[1:]))).reshape(3, 3)
            elif "R_rect_03" in name:
                R_rect_03 = np.array(list(map(float, cols[1:]))).reshape(3, 3)
    left_fx = P_rect_left[0, 0]
    left_fy = P_rect_left[1, 1]
    left_cx = P_rect_left[0, 2]
    left_cy = P_rect_left[1, 2]
    right_fx = P_rect_right[0, 0]
    right_fy = P_rect_right[1, 1]
    right_cx = P_rect_right[0, 2]
    right_cy = P_rect_right[1, 2]
    left_bx = P_rect_left[0, 3] / -left_fx
    right_bx = P_rect_right[0, 3] / -right_fx
    left_cam = CameraParameters(fx=left_fx, fy=left_fy, cx=left_cx, cy=left_cy)
    right_cam = CameraParameters(fx=right_fx, fy=right_fy, cx=right_cx, cy=right_cy)
    R_rect_23 = np.linalg.inv(R_rect_02) @ R_rect_03
    T23 = np.identity(4)
    # T23[:3, :3] = R_rect_23
    T23[0, 3] = right_bx - left_bx
    T02 = np.identity(4)
    T02[0, 3] = left_bx
    return StereoCamera(left_cam, right_cam, T23), T02


def get_estimated_obj_detections(
    kitti_path: Path, scene: str, side: str = "left"
) -> Dict[int, List[ObjectDetection]]:
    # adapted from https://github.com/VisualComputingInstitute/mots_tools/blob/master/mots_common/io.py
    objects_per_frame = {}
    combined_mask_per_frame = {}  # To check that no frame contains overlapping masks
    path = _get_estimated_detection_file(kitti_path, scene, side)
    with open(path.as_posix(), "r") as f:
        for line in f:
            line = line.strip()
            fields = line.split(" ")

            frame = int(fields[0])
            track_id = int(fields[1])
            class_id = int(fields[2])

            if frame not in objects_per_frame:
                objects_per_frame[frame] = []

            if class_id not in [1, 2]:
                continue
            cls = "car" if class_id == 1 else "pedestrian"
            w, h = map(int, fields[3:5])
            mask = {"size": [w, h], "counts": fields[5].encode(encoding="UTF-8")}

            if frame not in combined_mask_per_frame:
                combined_mask_per_frame[frame] = mask
            elif (
                rletools.area(
                    rletools.merge(
                        [combined_mask_per_frame[frame], mask], intersect=True
                    )
                )
                > 0.0
            ):
                pass
            else:
                combined_mask_per_frame[frame] = rletools.merge(
                    [combined_mask_per_frame[frame], mask], intersect=False
                )
            bool_mask = rletools.decode(mask).astype(bool)
            if bool_mask.sum() > 2:
                objects_per_frame[frame].append(
                    ObjectDetection(bool_mask, cls, track_id=track_id)
                )
    return objects_per_frame


def get_gt_obj_detections_from_kitti(
    kitti_path: Path, scene: str, img_id: int
) -> List[ObjectDetection]:
    instance_file = _get_instance_file(kitti_path, scene, str(img_id).zfill(6))
    img = np.array(cv2.imread(instance_file.as_posix(), cv2.IMREAD_ANYDEPTH))
    obj_ids = np.unique(img)
    obj_detections = []
    for obj_id in obj_ids:
        if obj_id in [0, 10000]:
            continue
        track_id = int(obj_id % 1000)
        obj_mask = img == obj_id

        class_id = int(obj_id // 1000)
        obj_class = "car" if class_id == 1 else "pedestrian"

        if obj_mask.sum() <= 2:
            continue
        obj_det = ObjectDetection(mask=obj_mask, track_id=track_id, cls=obj_class,)
        obj_detections.append(obj_det)
    return obj_detections


def _get_oxts_file(kitti_path: Path, scene: str) -> Path:
    return kitti_path / "oxts" / (scene + ".txt")


def _get_estimated_detection_file(kitti_path: Path, scene: str, side: str) -> Path:
    side_dir = "image_02" if side == "left" else "image_03"
    return kitti_path / "detections" / side_dir / (scene + ".txt")


def _get_calib_cam_to_cam_file(kitti_path: Path) -> Path:
    return kitti_path / "calib_cam_to_cam.txt"


def _get_calib_file(kitti_path: Path, scene: str) -> Path:
    return kitti_path / "calib" / (scene + ".txt")


def _get_instance_file(kitti_path: Path, scene: str, img_id: str) -> Path:
    return kitti_path / "instances" / scene / (img_id + ".png")


def _get_detection_file(kitti_path: Path, scene: str) -> Path:
    return kitti_path / "label_02" / (scene + ".txt")


def _oxts_to_poses(lat, lon, alt, roll, pitch, yaw):
    """This implementation is a python reimplementation of the convertOxtsToPose
    MATLAB function in the original development toolkit for raw data
    """

    def rot_x(theta):
        theta = np.atleast_1d(theta)
        n = len(theta)
        return np.stack(
            (
                np.stack([np.ones(n), np.zeros(n), np.zeros(n)], axis=-1),
                np.stack([np.zeros(n), np.cos(theta), -np.sin(theta)], axis=-1),
                np.stack([np.zeros(n), np.sin(theta), np.cos(theta)], axis=-1),
            ),
            axis=-2,
        )

    def rot_y(theta):
        theta = np.atleast_1d(theta)
        n = len(theta)
        return np.stack(
            (
                np.stack([np.cos(theta), np.zeros(n), np.sin(theta)], axis=-1),
                np.stack([np.zeros(n), np.ones(n), np.zeros(n)], axis=-1),
                np.stack([-np.sin(theta), np.zeros(n), np.cos(theta)], axis=-1),
            ),
            axis=-2,
        )

    def rot_z(theta):
        theta = np.atleast_1d(theta)
        n = len(theta)
        return np.stack(
            (
                np.stack([np.cos(theta), -np.sin(theta), np.zeros(n)], axis=-1),
                np.stack([np.sin(theta), np.cos(theta), np.zeros(n)], axis=-1),
                np.stack([np.zeros(n), np.zeros(n), np.ones(n)], axis=-1),
            ),
            axis=-2,
        )

    n = len(lat)

    # converts lat/lon coordinates to mercator coordinates using mercator scale
    #        mercator scale             * earth radius
    scale = np.cos(lat[0] * np.pi / 180.0) * 6378137

    position = np.stack(
        [
            scale * lon * np.pi / 180.0,
            scale * np.log(np.tan((90.0 + lat) * np.pi / 360.0)),
            alt,
        ],
        axis=-1,
    )

    R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)

    # extract relative transformation with respect to the first frame
    T0_inv = np.block([[R[0].T, -R[0].T @ position[0].reshape(3, 1)], [0, 0, 0, 1]])
    T = T0_inv @ np.block(
        [[R, position[:, :, None]], [np.zeros((n, 1, 3)), np.ones((n, 1, 1))]]
    )
    return T
