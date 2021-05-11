import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb
from bamot.obbox_regressor.dataloader import BAMOTPointCloudDataModule
from bamot.obbox_regressor.model import OBBoxRegressor


def get_trainer(args, wandb_logger):
    return pl.Trainer(
        logger=wandb_logger,
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        callbacks=[pl.callbacks.EarlyStopping("avg_val_loss")],
    )


def main(args):
    model = OBBoxRegressor(
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        use_gpus=args.gpus is not None,
        encode_pointcloud=args.encode_pointcloud,
        num_points=args.num_points,
    )
    dm = BAMOTPointCloudDataModule(
        dataset_dir=args.input_files_dir,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        pointcloud_size=args.num_points,
    )
    wandb.finish()
    wandb_logger = WandbLogger(
        project="obbox-regressor", entity="tum-3dmot", offline=args.gpus is None
    )
    if args.multi_stage:
        # multi-stage training
        model.train_target = "dim"
        trainer = get_trainer(args, wandb_logger)
        trainer.fit(model, dm)

        model.train_target = "pos"
        trainer = get_trainer(args, wandb_logger)
        trainer.fit(model, dm)

        model.train_target = "dim"
        trainer = get_trainer(args, wandb_logger)
        trainer.fit(model, dm)
    else:
        model.train_target = "all"
        trainer = get_trainer(args, wandb_logger)
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
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--multi-stage", action="store_true")
    args = parser.parse_args()
    main(args)
