import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb
from bamot.obbox_regressor.dataloader import BAMOTPointCloudDataModule
from bamot.obbox_regressor.model import OBBoxRegressor


def main(args):
    model = OBBoxRegressor(
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        use_gpus=args.gpus is not None,
        encode_pointcloud=args.encode_pointcloud,
    )
    dm = BAMOTPointCloudDataModule(
        dataset_dir=args.input_files_dir,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
    )
    wandb.finish()
    wandb_logger = WandbLogger(project="obbox-regressor", entity="tum-3dmot")
    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        precision=args.precision,
    )
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)


if __name__ == "__main__":
    pl.seed_everything(42)
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("input_files_dir", type=str)
    parser.add_argument(
        "--max-epochs", type=int, default=5, help="Max epochs (default 5)"
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=8)
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-7)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--encode-pointcloud", default=False)
    parser.add_argument("--precision", choices=[16, 32], default=32)
    args = parser.parse_args()
    main(args)
