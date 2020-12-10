import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from bamot.config import CONFIG as config
from bamot.util.kitti import get_gt_poses_from_kitti, get_label_data_from_kitti
from torch.utils.data import DataLoader, Dataset, random_split


class BAMOTPointCloudDataset(Dataset):
    def __init__(
        self, dataframe: pd.DataFrame, pointcloud_size: int, **kwargs,
    ):
        super().__init__(**kwargs)
        self._dataframe = dataframe
        self._pointcloud_size = pointcloud_size
        self._rng = np.random.default_rng(42)

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
        return pointcloud.T

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self._dataframe.iloc[idx]
        ptc_fname = row.pointcloud_fname
        num_poses = row.num_poses
        num_other_tracks = row.num_other_tracks
        feature_vector = torch.Tensor(np.array([num_poses, num_other_tracks]))
        target_vector = torch.Tensor(row.target.tolist())
        # read pointcloud and convert to tensor
        pointcloud = torch.Tensor(self._load_and_process_pointcloud(ptc_fname))

        return dict(
            pointcloud=pointcloud, target=target_vector, feature_vector=feature_vector
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
        self._dataset_dir = dataset_dir
        self._track_id_mapping = track_id_mapping
        self._train_val_test_ratio = train_val_test_ratio
        self._pointcloud_size = pointcloud_size
        self._train_batch_size = train_batch_size
        self._eval_batch_size = eval_batch_size

    def setup(self, stage: str):
        all_files = list(
            filter(lambda f: f.suffix == ".csv", Path(self._dataset_dir).iterdir())
        )
        if not all_files:
            raise ValueError(f"No `.csv` files found at `{self._dataset_dir}`")
        dataset = None
        for f in all_files:
            df = pd.read_csv(f)
            if dataset is None:
                dataset = df
            else:
                dataset = dataset.append(df)
        dataset.dropna(inplace=True)

        # get all gt data for all scenes
        all_gt_data = {}
        for scene in range(21):
            gt_poses = get_gt_poses_from_kitti(
                kitti_path=config.KITTI_PATH, scene=scene
            )
            label_data = get_label_data_from_kitti(
                kitti_path=config.KITTI_PATH, scene=scene, poses=gt_poses
            )
            all_gt_data[scene] = label_data

        target_vectors = []
        for row in dataset.itertuples():
            scene = row.scene
            img_id = row.img_id
            if self._track_id_mapping:
                track_id = self._track_id_mapping.get(row.track_id)
                if track_id is None:
                    continue
            else:
                track_id = row.track_id
            row_data = all_gt_data[scene][track_id][img_id]
            target_vector = np.array(
                [*row_data.cam_pos, row_data.rot_angle, *row_data.dim_3d]
            ).reshape(-1)
            target_vectors.append(target_vector)
        dataset["target"] = target_vectors
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
        self._dataset = {}
        self._dataset["train"] = dataset.iloc[:train_size]
        self._dataset["val"] = dataset.iloc[train_size : train_size + val_size]
        self._dataset["test"] = dataset.iloc[-test_size:]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            BAMOTPointCloudDataset(
                self._dataset["train"], pointcloud_size=self._pointcloud_size
            ),
            batch_size=self._train_batch_size,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            BAMOTPointCloudDataset(
                self._dataset["val"], pointcloud_size=self._pointcloud_size
            ),
            batch_size=self._eval_batch_size,
            num_workers=os.cpu_count(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            BAMOTPointCloudDataset(
                self._dataset["test"], pointcloud_size=self._pointcloud_size
            ),
            batch_size=self._eval_batch_size,
            num_workers=os.cpu_count(),
        )
