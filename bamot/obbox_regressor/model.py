from typing import NamedTuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F


class NewsFeedAnalyzer(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 2e-5,
        train_batch_size: int = 2,
        eval_batch_size: int = 2,
        prob_dropout: float = 0.2,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self._backbone = None
        self._regressor = nn.Sequential(
            nn.Linear(dim_backbone, 128),
            nn.ReLU(),
            nn.Dropout(prob_dropout),
            nn.Linear(128, 7),
        )
