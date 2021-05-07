import argparse

import pytorch_lightning as pl
from bamot.obbox_regressor.dataloader import BAMOTPointCloudDataModule
from bamot.obbox_regressor.model import OBBoxRegressor
from pytorch_lightning.loggers import WandbLogger


def main(args):
    model = OBBoxRegressor(
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        on_gpu=args.gpus is not None,
    )
    dm = BAMOTPointCloudDataModule(
        dataset_dir=args.input_files_dir,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
    )
    wandb_logger = WandbLogger(project="bamot", offline=True)
    trainer = pl.Trainer(logger=wandb_logger, gpus=args.gpus)
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)


if __name__ == "__main__":
    data = "/Users/anselm/uni/ss2020/dynamic_slam/data/KITTI/tracking/training/obb_data/test"
    pl.seed_everything(42)
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("input_files_dir", type=str)
    parser.add_argument(
        "--max-epochs", type=int, default=5, help="Max epochs (default 5)"
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=8)
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-5)
    parser.add_argument("--gpus", default=None)
    args = parser.parse_args()
    main(args)
