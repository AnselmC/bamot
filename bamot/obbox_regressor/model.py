from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb as wb
from bamot.thirdparty.pointnet2.pointnet2.models.pointnet2_ssg_sem import \
    PointNet2SemSegSSG
from bamot.util.cv import get_corners_from_vector
from bamot.util.misc import get_color
from torch.nn import functional as F


class OBBoxRegressor(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 2e-5,
        num_points: int = 1024,
        train_batch_size: int = 2,
        eval_batch_size: int = 2,
        prob_dropout: float = 0.2,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self._backbone = PointNet2SemSegSSG(hparams={"model.use_xyz": True})
        dim_backbone = 13 * num_points
        self._regressor = nn.Sequential(
            nn.Linear(dim_backbone, 128),
            nn.ReLU(),
            nn.Dropout(prob_dropout),
            nn.Linear(128, 7),
        )
        self._est_color = get_color(normalized=True, as_tuple=True)
        self._gt_color = get_color(normalized=True, as_tuple=True)

    def forward(self, pointcloud, feature_vector):
        x = self._backbone(pointcloud)
        stacked = torch.stack(x.view(-1), feature_vector)
        out = self._regressor(stacked)
        return out

    def _get_location_loss(
        self, loc: torch.Tensor, target_loc: torch.Tensor
    ) -> torch.Tensor:
        return F.smooth_l1_loss(loc, target_loc)

    def _get_size_loss(
        self, size: torch.Tensor, target_size: torch.Tensor
    ) -> torch.Tensor:
        return F.smooth_l1_loss(size, target_size)

    def _get_angle_loss(
        self, angle: torch.Tensor, target_angle: torch.Tensor
    ) -> torch.Tensor:
        scaled_angle = torch.remainder(angle, torch.Tensor(np.pi))
        return F.smooth_l1_loss(scaled_angle, target_angle)

    def _get_losses(self, y: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor]:
        loc = y[:, :, :3]
        target_loc = target[:, :, :3]
        angle = y[:, :, 3:4]
        target_angle = target[:, :, 3:4]
        size = y[:, :, 4:]
        target_size = y[:, :, 4:]
        loc_loss = self._get_location_loss(loc, target_loc)
        angle_loss = self._get_angle_loss(angle, target_angle)
        size_loss = self._get_size_loss(size, target_size)
        return loc_loss, angle_loss, size_loss

    def training_step(self, batch, batch_idx):
        pointcloud = batch["pointcloud"]
        feature_vector = batch["feature_vector"]
        target = batch["target"]
        # size: B x N x 7
        y = self(pointcloud, feature_vector)
        loc_loss, angle_loss, size_loss = self._get_losses(y, target)
        self.log("loc_loss_train", loc_loss)
        self.log("angle_loss_train", angle_loss)
        self.log("size_loss_train", size_loss)
        total_loss = loc_loss + angle_loss + size_loss
        self.log("loss_train", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        pointcloud = batch["pointcloud"]
        feature_vector = batch["feature_vector"]
        target = batch["target"]
        y = self(pointcloud, feature_vector)
        loc_loss, angle_loss, size_loss = self._get_losses(y, target)
        self.log("loc_loss_valid", loc_loss)
        self.log("angle_loss_valid", angle_loss)
        self.log("size_loss_valid", size_loss)
        total_loss = loc_loss + angle_loss + size_loss
        self.log("loss_valid", total_loss)
        # log first pointcloud + GT box + Est Box from batch
        gt_corners = get_corners_from_vector(target[0])
        est_corners = get_corners_from_vector(y[0])
        self.log(
            "estimated_obbox",
            {
                "point_scene": wb.Object3D(
                    {
                        "type": "lidar/beta",
                        "points": pointcloud[0],
                        "boxes": np.array(
                            [
                                {
                                    "corners": gt_corners,
                                    "label": "GT OBBox",
                                    "color": self._gt_color,
                                },
                                {
                                    "corners": est_corners,
                                    "label": "EST OBBox",
                                    "color": self._est_color,
                                },
                            ]
                        ),
                    }
                )
            },
        )

    def test_step(self, batch, batch_idx):
        pointcloud = batch["pointcloud"]
        feature_vector = batch["feature_vector"]
        target = batch["target"]
        y = self(pointcloud, feature_vector)
        loc_loss, angle_loss, size_loss = self._get_losses(y, target)
        self.log("loc_loss_test", loc_loss)
        self.log("angle_loss_test", angle_loss)
        self.log("size_loss_test", size_loss)
        total_loss = loc_loss + angle_loss + size_loss
        self.log("loss_test", total_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
