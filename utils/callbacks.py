import lightning as L

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import TQDMProgressBar

class ModProgressBar(TQDMProgressBar):
    '''Slightly modified version of ProgressBar to remove v_num entry, see
    https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ProgressBar.html'''

    def __init__(self):
        super().__init__()
        self.enable = True

    def disable(self):
        self.enable = False

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

def enable_callbacks(cfg):
    # Define callbacks
    callbacks = []
    callbacks.append(ModProgressBar())

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

    if cfg.training.checkpointing.enabled:
        # Keep the last 10 checkpoints
        callbacks.append(
            ModelCheckpoint(
                filename="{epoch:02d}",
                monitor="step",
                mode="max",
                save_top_k=10,
                save_last=True,
                every_n_epochs=1,
                save_on_train_epoch_end=True,
            )
        )

        # Keep only the best checkpoint
        callbacks.append(
            ModelCheckpoint(
                filename="best",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            )
        )
    return callbacks
