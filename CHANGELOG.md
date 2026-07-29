# Changelog

## v1.02
### Dataloader
- Improved robustness by ensuring consistency in:
  - Latitude/longitude matching across years
  - Grid orientation
  - Variable ordering
- Separated `getitem` logic for prediction versus training/validation to reduce inference cost.
- Added support for configuring different numbers of workers for training and validation.

### Model
- Reformulated the model as a residual model.
- Improved near-surface fields by adding a static feature encoder in the reaction term.
- Added support for performing diffusion at a smaller resolution.

### Inference
- Added multi-GPU inference support.
- Reduced inference-mode memory usage with a `flush-every-n-steps` argument.

### Other
- Switched output normalization to use the standard deviation of 6-hour values.
- Added support for Muon and Normuon optimizers.
- Updated data preprocessing to rely on the LayerQuantier compressor.

## v1.03
### Model
- Changed the model to predict non-residual outputs.
- Updated the default hyperparameters in the configuration file.
- Enabled a coarsened latent physics processor.
- Switched training to manual optimization to support customized fine-tuning behavior.

### Inference
- Added an option for configuring the number of data-loading workers.

### Other
- Disabled tendency normalization when using non-residual outputs.
-Performed general code cleanup and removed stale code.
