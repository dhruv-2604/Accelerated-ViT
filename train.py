"""
Training Entry Point for Accelerated ViT

This is the main script that:
1. Loads configuration via Hydra
2. Sets up the DataModule and Model
3. Configures logging (WandB)
4. Runs training via PyTorch Lightning

Usage:
    # Default training
    python train.py

    # With overrides
    python train.py data.batch_size=256 training.learning_rate=0.001

    # Load experiment preset
    python train.py experiment=tiny_vit_cuda

    # PACE ICE (called from train.sh)
    python train.py experiment=tiny_vit_cuda data.data_dir=$TMPDIR/tiny-imagenet-200
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from src.data.tiny_imagenet import TinyImageNetDataModule
from src.system import ViTLightningModule


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main training function.

    Args:
        cfg: Hydra configuration (automatically loaded from YAML files)
    """
    # ========================================================================
    # Print configuration (helpful for debugging)
    # ========================================================================
    print("=" * 70)
    print("Configuration:")
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 70)

    # ========================================================================
    # Set random seed for reproducibility
    # ========================================================================
    pl.seed_everything(42, workers=True)

    # ========================================================================
    # Create DataModule
    # ========================================================================
    print("\n📦 Setting up data...")
    datamodule = TinyImageNetDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    # ========================================================================
    # Create Model
    # ========================================================================
    print("\n🧠 Creating ViT model...")
    model = ViTLightningModule(
        # Model architecture
        img_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        embed_dim=cfg.model.embed_dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout=cfg.model.dropout,
        # Training hyperparameters
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_epochs=cfg.training.warmup_epochs,
        max_epochs=cfg.training.max_epochs,
    )

    # Print model summary
    print(f"   Embed dim: {cfg.model.embed_dim}")
    print(f"   Depth: {cfg.model.depth}")
    print(f"   Heads: {cfg.model.num_heads}")

    # ========================================================================
    # Setup Callbacks
    # ========================================================================
    callbacks = [
        # Save best model based on validation accuracy
        ModelCheckpoint(
            monitor="val/acc",
            mode="max",
            save_top_k=1,
            filename="best-{epoch:02d}-{val_acc:.3f}",
        ),
        # Log learning rate (helpful for debugging schedule)
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ========================================================================
    # Setup Logger (Weights & Biases)
    # ========================================================================
    print("\n📊 Setting up WandB logger...")
    logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        tags=cfg.wandb.tags,
        log_model=True,  # Log model checkpoints to WandB
    )

    # ========================================================================
    # Create Trainer
    # ========================================================================
    print("\n🚀 Creating trainer...")
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        max_epochs=cfg.trainer.max_epochs,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        enable_checkpointing=cfg.trainer.enable_checkpointing,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
    )

    # ========================================================================
    # Train!
    # ========================================================================
    print("\n🎯 Starting training...")
    print("=" * 70)
    trainer.fit(model, datamodule)

    # ========================================================================
    # Print final results
    # ========================================================================
    print("\n" + "=" * 70)
    print("✅ Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
