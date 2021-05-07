from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from bamot.config import CONFIG as config
from bamot.util.kitti import (get_gt_detection_data_from_kitti,
                              get_gt_poses_from_kitti)
from torch.utils.data import DataLoader, Dataset, random_split


class BAMOTPointCloudDataset(Dataset):
    def __init__(
        self, dataframe: pd.DataFrame, pointcloud_size: int, base_path: Path, **kwargs,
    ):
        super().__init__(**kwargs)
        self._dataframe = dataframe
        self._pointcloud_size = pointcloud_size
        self._rng = np.random.default_rng(42)
        self._base_path = base_path

    def __len__(self):
        return len(self._dataframe)

    def _load_and_process_pointcloud(self, pointcloud_fname):
        pointcloud = np.load(pointcloud_fname).reshape(3, -1).astype(np.float32)
        if len(pointcloud.T) != self._pointcloud_size:
            # randomly drop or repeat points
            pointcloud = self._rng.choice(
                pointcloud,
                size=self._pointcloud_size,
                replace=len(pointcloud) < self._pointcloud_size,
                axis=1,
            )
        return pointcloud

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self._dataframe.iloc[idx]
        feature_vector = torch.Tensor(
            np.array(
                [
                    row.num_poses,
                    row.num_points,
                    row.dist_from_cam,
                    row.badly_tracked_frames,
                ]
            )
        )
        target_yaw = torch.Tensor([row.target_yaw])
        target_pos = torch.Tensor(row.target_pos)
        # read pointcloud and convert to tensor
        ptc_fname = (
            self._base_path
            / str(row.scene).zfill(4)
            / f"{int(row.img_id)}_{int(row.track_id)}.npy"
        )
        pointcloud = torch.Tensor(self._load_and_process_pointcloud(ptc_fname))

        return dict(
            pointcloud=pointcloud,
            feature_vector=feature_vector,
            target_yaw=target_yaw,
            target_pos=target_pos,
            est_yaw=torch.Tensor([row.yaw]),
            est_pos=torch.Tensor([row.x, row.y, row.z]),
        )


class BAMOTPointCloudDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        train_val_test_ratio: Tuple[int, int, int] = (8, 1, 1),
        track_id_mapping: Dict[int, int] = {},
        pointcloud_size: int = 1024,
        train_batch_size: int = 2,
        eval_batch_size: int = 2,
        **kwargs,
    ):
        super().__init__()
        self._dataset_dir = Path(dataset_dir)
        self._track_id_mapping = track_id_mapping
        self._train_val_test_ratio = train_val_test_ratio
        self._pointcloud_size = pointcloud_size
        self._train_batch_size = train_batch_size
        self._eval_batch_size = eval_batch_size
        self._dataset: Dict = {}

    def setup(self, stage: Optional[str] = None):
        all_files = list(
            filter(lambda f: f.suffix == ".csv", self._dataset_dir.iterdir())
        )
        if not all_files:
            raise ValueError(f"No `.csv` files found at `{self._dataset_dir}`")
        first_file = True
        for f in all_files:
            df = pd.read_csv(f)
            if first_file:
                dataset = df
                first_file = False
            else:
                dataset = dataset.append(df)
        dataset.dropna(inplace=True)

        target_yaw = []
        target_pos = []
        all_gt_data = {}
        idx_to_remove = []
        for idx, row in enumerate(dataset.itertuples()):
            scene = int(row.scene)
            if scene not in all_gt_data:  # load gt data dynamically
                gt_poses = get_gt_poses_from_kitti(
                    kitti_path=config.KITTI_PATH, scene=scene
                )
                label_data = get_gt_detection_data_from_kitti(
                    kitti_path=config.KITTI_PATH, scene=scene, poses=gt_poses
                )
                all_gt_data[scene] = label_data

            img_id = int(row.img_id)
            if self._track_id_mapping:
                track_id = self._track_id_mapping.get(row.track_id)
                if track_id is None:
                    idx_to_remove.append(idx)
                    continue
            else:
                track_id = int(row.track_id)
            gt_track_data = all_gt_data[scene].get(track_id)
            if gt_track_data is None:
                idx_to_remove.append(idx)
                continue
            gt_data = gt_track_data.get(img_id)
            if gt_data is None:
                idx_to_remove.append(idx)
                continue  # no corresponding GT detection
            target_yaw.append(gt_data.rot_angle)
            target_pos.append(gt_data.cam_pos)
        dataset.drop(idx_to_remove, inplace=True)
        dataset["target_yaw"] = target_yaw
        dataset["target_pos"] = target_pos
        size = len(dataset)
        val_size = int(
            size * (self._train_val_test_ratio[1] / sum(self._train_val_test_ratio))
        )
        test_size = int(
            size * (self._train_val_test_ratio[2] / sum(self._train_val_test_ratio))
        )
        train_size = size - val_size - test_size
        # shuffle dataframe first
        dataset = dataset.sample(frac=1, random_state=42)
        self._dataset["train"] = dataset.iloc[:train_size]
        self._dataset["val"] = dataset.iloc[train_size : train_size + val_size]
        self._dataset["test"] = dataset.iloc[-test_size:]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            BAMOTPointCloudDataset(
                self._dataset["train"],
                pointcloud_size=self._pointcloud_size,
                base_path=self._dataset_dir,
            ),
            batch_size=self._train_batch_size,
            num_workers=1,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            BAMOTPointCloudDataset(
                self._dataset["val"],
                pointcloud_size=self._pointcloud_size,
                base_path=self._dataset_dir,
            ),
            batch_size=self._eval_batch_size,
            num_workers=1,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            BAMOTPointCloudDataset(
                self._dataset["test"],
                pointcloud_size=self._pointcloud_size,
                base_path=self._dataset_dir,
            ),
            batch_size=self._eval_batch_size,
            num_workers=1,
        )
