import lightning as L

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import TQDMProgressBar


class ModProgressBar(TQDMProgressBar):
    """Slightly modified version of ProgressBar to remove v_num entry, see
    https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ProgressBar.html
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable()

    def disable(self):
        super().disable()

    def get_metrics(self, trainer, pl_module):
        # don't show the version number
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items

    def on_train_epoch_start(self, trainer, *_):
        # Default: replaces the training progress bar with one
        # for the current epoch.  Modified: *don't* replace
        # the progress bar, and ensure it's set for a total
        # number of training steps
        total_batches = trainer.estimated_stepping_batches
        # Calculate max epochs
        if trainer.max_epochs is not None and trainer.max_epochs > 0:
            max_epochs = trainer.max_epochs
        else:
            max_epochs = 1 + (total_batches - 1) // trainer.num_training_batches
        if total_batches != self.train_progress_bar.total:
            # Store the current progress bar info
            n = self.train_progress_bar.n
            last_print_n = self.train_progress_bar.last_print_n
            last_print_t = self.train_progress_bar.last_print_t
            start_t = self.train_progress_bar.start_t
            # Reset the progress bar total
            self.train_progress_bar.reset(total_batches)
            # Restore current n
            self.train_progress_bar.update(n)
            # Restore printing settings so progress isn't screwed up
            self.train_progress_bar.last_print_n = last_print_n
            self.train_progress_bar.last_print_t = last_print_t
            self.train_progress_bar.start_t = start_t
        self.train_progress_bar.set_description(
            f"Epoch {trainer.current_epoch+1}/{max_epochs}"
        )
        self.train_progress_bar.refresh()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Update the progress bar based on the total number of batches
        n = int(batch_idx + trainer.current_epoch * trainer.num_training_batches + 1)
        if self._should_update(n, self.train_progress_bar.total):
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
            if not (self.train_progress_bar.disable):
                self.train_progress_bar.n = n
                self.train_progress_bar.refresh()

    def on_train_epoch_end(self, trainer, pl_module):
        # Override default behaviour to not close the progress
        # bar at epoch-end, regardless of 'leave' parameter (the bar isn't done)
        if not self.train_progress_bar.disable:
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_train_end(self, *_):
        # Explicitly close the *container* of the progress bar, working around the
        # inexplicable littering of progress bars in Jupyter notebooks

        # First, close validation progress bar if it's still open
        if self._val_progress_bar is not None:
            # print(f'val_progress_bar is {repr(self._val_progress_bar)}')
            if "container" in self.val_progress_bar.__dict__:
                self.val_progress_bar.container.close()
            self.val_progress_bar.close()

        if not self._leave and "container" in self.train_progress_bar.__dict__:
            self.train_progress_bar.container.close()
        return super().on_train_end(*_)

    def on_validation_start(self, trainer, pl_module):
        # Redefine on_validation_start to only create a progress bar if
        # one does not exist
        if not trainer.sanity_checking:
            # print(f'val_progress_bar is {repr(self._val_progress_bar)}')
            if self._val_progress_bar is None or self.val_progress_bar.disable:
                self.val_progress_bar = self.init_validation_tqdm()
            # else:
            #     raise ValueError()

    def on_validation_end(self, trainer, pl_module):
        """
        Replicate TQDMProgressBar.on_validation_end, except keep the validation progress bar open
        if we're inside a Jupyter notebook.  Otherwise, repeated close/reopen of the progress bar
        can confuse the notebook cell into thinking it's had too much output, blocking further
        training/validation progress bar updates
        """

        if (
            self._val_progress_bar is not None
            and "container" not in self.val_progress_bar.__dict__
        ):
            self.val_progress_bar.close()

        self.reset_dataloader_idx_tracker()
        if self._train_progress_bar is not None and trainer.state.fn == "fit":
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))


def enable_callbacks(cfg):
    # Define callbacks
    callbacks = []
    if cfg.training.progress_bar and not cfg.training.print_losses:
        callbacks.append(ModProgressBar(leave=False))

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
