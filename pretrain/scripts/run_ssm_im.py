import os
import torch
import argparse

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from llmlib.architectures.models.mamba import get_model
from llmlib.data.image import load_cifar, dataloaders


def main(args):
    L.seed_everything(args.seed)
    model_name = "".join(f"{k[:2]}{v}" for k, v in vars(args).items())
    checkpoint_path = f"checkpoints/{model_name}"
    os.makedirs(checkpoint_path, exist_ok=True)


    wandb_logger = L.pytorch.loggers.WandbLogger(
        project="ssm-cifar-tokenized",
        notes="testing out ssms on tokenized cifar",
        tags=["ssm", "cifar"],
        config=args,
    )

    data = load_cifar()
    # set vocab size for mamba based on dataset
    args.vocab_size = data.d_output

    train_loader, valid_loader, test_loader = dataloaders(
        data, args.batch_size, args.num_workers
    )
    model = get_model(args)

    val_chp = ModelCheckpoint(
        save_top_k=2,
        mode="min",
        monitor="valid/bpd",
        dirpath=checkpoint_path,
        filename="cifar-{epoch:02d}-{val_bpd:.2f}",
    )
    trainer = L.Trainer(
        default_root_dir=checkpoint_path,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count(),
        strategy="ddp",
        max_epochs=args.num_epochs,
        max_steps=args.train_steps,
        accumulate_grad_batches=args.grad_accumulation_steps,
        gradient_clip_val=args.grad_clip_val,
        callbacks=[
            val_chp,
            LearningRateMonitor("epoch"),
        ],
        num_sanity_val_steps=5,
        logger = wandb_logger,
    )

    trainer.fit(model, train_loader, valid_loader)
    # don't need to rerun validation
    #valid_result = trainer.validate(model, valid_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)


if __name__ == "__main__":
    # TODO replace parser with hydra
    parser = argparse.ArgumentParser(description="Train a model on CIFAR with SSM")
    parser.add_argument(
        "--model",
        choices=["MambaSubPixelLm", "MambaPixelLm"],
        default="MambaSubPixelLm",
        help="AR Model (default: MambaSubPixelLm)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for training (default: 50)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--d-model", type=int, default=512, help="Dimension of model (default: 512)"
    )
    parser.add_argument(
        "--n-layer", type=int, default=16, help="Number of layers (default: 16)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=200, help="Number of epochs (default: 200)"
    )
    parser.add_argument(
        "--patience", type=int, default=4, help="Number of plateau epochs (default: 4)"
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=0.5,
        help="Decay factor after plateau (default: 0.5)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Number of samples at eval (default: 16)",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=-1,
        help="Number of batches to run in training. Debugging only. (default: -1)",
    )
    parser.add_argument(
        "--grad-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 2)",
    )
    parser.add_argument(
        "--grad-clip-val",
        type=float,
        default=1.0,
        help="Gradient accumulation steps (default: 1)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Gradient accumulation steps (default: 0.2)",
    )
    parser.add_argument(
        "--save-model-steps",
        type=int,
        default=500,
        help="Save model every n steps (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Save model every n steps (default: 1234)",
    )

    args = parser.parse_args()

    main(args)
