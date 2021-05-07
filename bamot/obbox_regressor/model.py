from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from bamot.config import CONFIG as config
from bamot.util.cv import get_corners_from_vector, get_oobbox_vec
from bamot.util.misc import get_color
from bamot.util.viewer import Colors, visualize_pointcloud_and_obb
from torch.nn import functional as F

import wandb as wb


@dataclass
class Stages:
    TRAIN: str = "train"
    VAL: str = "val"
    TEST: str = "test"


class OBBoxRegressor(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 2e-7,
        num_points: int = 1024,
        train_batch_size: int = 2,
        eval_batch_size: int = 2,
        prob_dropout: float = 0.2,
        dim_feature_vector: int = 4,
        log_pcl_every_n_steps: int = 40,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        # pointnet2 not available for CPU, but need to be able to run for sanity check of pipeline
        if torch.cuda.is_available():
            from bamot.thirdparty.pointnet2.pointnet2.models.pointnet2_ssg_sem import \
                PointNet2SemSegSSG

            self._backbone = nn.Sequential(
                PointNet2SemSegSSG(hparams={"model.use_xyz": True}), nn.Flatten()
            )
            # feature vector has length 13
            dim_backbone = 13 * num_points + dim_feature_vector
            self._device = "cuda"
        else:
            self._backbone = nn.Flatten()
            # x,y,z coordinates
            dim_backbone = 3 * num_points + dim_feature_vector
            self._device = "cpu"
        self._regressor = nn.Sequential(
            nn.Linear(dim_backbone, 128),
            nn.ReLU(),
            nn.Dropout(prob_dropout),
            nn.Linear(128, 4),  # predict position and yaw corrections
        )
        self._stage = None
        self._log_pcl_every_n_steps = log_pcl_every_n_steps

    def forward(self, pointcloud, feature_vector):
        # pointcloude shape: B x N x 3
        x = self._backbone(pointcloud)
        # stack feature vector + flattened output from pointnet2
        stacked = torch.cat((x, feature_vector), 1)
        out = self._regressor(stacked)
        return out

    def _get_location_loss(
        self, loc: torch.Tensor, target_loc: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(loc, target_loc)

    def _get_size_loss(
        self, size: torch.Tensor, target_size: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(size, target_size)

    def _get_angle_loss(
        self, angle: torch.Tensor, target_angle: torch.Tensor
    ) -> torch.Tensor:
        scaled_angle = torch.remainder(angle, torch.Tensor([np.pi]).to(self._device))
        return F.mse_loss(scaled_angle, target_angle)

    def _generic_step(self, batch, batch_idx):
        pcl = batch["pointcloud"]
        fv = batch["feature_vector"]

        est_yaw = batch["est_yaw"]
        est_pos = batch["est_pos"]

        target_yaw = batch["target_yaw"]
        target_pos = batch["target_pos"]

        # regress against correction
        target_yaw_residual = target_yaw - est_yaw
        target_pos_residual = target_pos - est_pos
        # size: B x N x 4 (3 for pos, 1 for yaw)
        preds = self(pcl, fv)
        pos = preds[:, :3]
        yaw = preds[:, 3:4]
        # compute losses
        pos_loss = self._get_location_loss(pos, target_pos_residual)
        yaw_loss = self._get_angle_loss(yaw, target_yaw_residual)
        self.log(
            f"pos_loss_{self._stage}", pos_loss, on_epoch=self._stage == Stages.TRAIN
        )
        self.log(
            f"yaw_loss_{self._stage}", yaw_loss, on_epoch=self._stage == Stages.TRAIN
        )
        total_loss = pos_loss + yaw_loss
        self.log(
            f"total_loss_{self._stage}",
            total_loss,
            on_epoch=self._stage == Stages.TRAIN,
        )

        if (batch_idx % self._log_pcl_every_n_steps) == 0:
            # log first pointcloud + GT box + Est Box from batch
            # expected vector: x, y, z, theta, height, width, length = vec.reshape(7, 1)
            target_vec = get_oobbox_vec(
                pos=target_pos[0].cpu().detach().numpy(),
                yaw=target_yaw[0].cpu().detach().numpy(),
                dims=config.CAR_DIMS,
            )
            gt_corners = get_corners_from_vector(target_vec)
            init_est_vec = get_oobbox_vec(
                pos=est_pos[0].cpu().detach().numpy(),
                yaw=est_yaw[0].cpu().detach().numpy(),
                dims=config.CAR_DIMS,
            )
            init_est_corners = get_corners_from_vector(init_est_vec)
            est_vec = get_oobbox_vec(
                pos=(est_pos + pos)[0].cpu().detach().numpy(),
                yaw=(est_yaw + yaw)[0].cpu().detach().numpy(),
                dims=config.CAR_DIMS,
            )

            est_corners = get_corners_from_vector(est_vec)

            pcl_cam = (pcl.T + est_pos.T).T
            # visualize_pointcloud_and_obb(
            #    pcl_cam[0].numpy(),
            #    [gt_corners, init_est_corners, est_corners],
            #    colors=[Colors.WHITE, Colors.RED, Colors.GREEN],
            # )
            wb.log(
                {
                    "point_scene": wb.Object3D(
                        {
                            "type": "lidar/beta",
                            "points": pcl_cam[0].cpu().detach().numpy(),
                            "boxes": np.array(
                                [
                                    {
                                        "corners": gt_corners.T.tolist(),
                                        "label": "GT OBBox",
                                        "color": Colors.WHITE,
                                    },
                                    {
                                        "corners": est_corners.T.tolist(),
                                        "label": "Est. OBBox",
                                        "color": Colors.GREEN,
                                    },
                                    {
                                        "corners": init_est_corners.T.tolist(),
                                        "label": "Initial est. OBBox",
                                        "color": Colors.RED,
                                    },
                                ]
                            ),
                        }
                    )
                },
            )
        return total_loss

    def training_step(self, batch, batch_idx):
        self._stage = Stages.TRAIN
        return self._generic_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self._stage = Stages.VAL
        self._generic_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self._stage = Stages.TEST
        self._generic_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
