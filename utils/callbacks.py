from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

def enable_callbacks(cfg):
    # Define callbacks
    callbacks = []
    if cfg.training.early_stopping.enabled:
        # Stop epochs when validation loss is not decreasing during a coupe of epochs
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=cfg.training.early_stopping.patience,
                check_finite=True,  # Make sure validation has not gone to nan
            )
        )

    # Keep the last 10 checkpoints and the top "best" checkpoint

    if cfg.training.checkpointing.enabled:
        callbacks.append(
            ModelCheckpoint(
                filename="{epoch}",  # Filename format for the checkpoints
                monitor="train_loss",
                save_top_k=10,  # Keep the last 10 checkpoints
                save_last=True,  # Always save the most recent checkpoint
                every_n_epochs=1,  # Save at the end of every epoch
            )
        )

        callbacks.append(
            ModelCheckpoint(
                filename="best",
                monitor="val_loss",
                mode="min",
                save_top_k=1,  # Keep only the best checkpoint
            )
        )
    return callbacks