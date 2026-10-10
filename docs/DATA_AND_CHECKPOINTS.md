# Data and model checkpoints

The repository should contain only data that can legally and practically be redistributed. Do not commit a complete training dataset, private experimental data, generated output folders, or model weights by default.

## Data-list files

Training, validation, and test list files use three whitespace-separated paths per line:

```text
relative/path/to/WF.tif relative/path/to/SIM.tif relative/path/to/PSF.tif
```

These paths are resolved under the `--datapath` root. The training, validation, and test lists may refer to different sample sets. Keep the existing paper split intact when reproducing published results.

## Public test examples

If `Images_for_test/` is included, document which files are WF images, reconstructed SIM images, raw SIM acquisitions, PSFs, and optional GT/reference data. State the source and redistribution permission for each dataset. Include only a small representative sample set that is authorized for public sharing.

## Checkpoints

Model checkpoints can be large and can encode architecture-specific parameters. If a trained checkpoint is distributed:

1. Identify the model architecture and any ablation settings used to train it.
2. State which script or GUI model can load it.
3. State any license or use restrictions that apply to the checkpoint.
4. Provide a stable versioned download location if the file is too large for the Git repository.

When no checkpoint is distributed, make that explicit and document how a user can train a compatible model. Never publish a checkpoint simply because it exists in a local `checkpoint/` directory.

## Generated files

Training logs, TensorBoard events, temporary files, quantitative-analysis tables, difference images, and inference outputs should be generated locally. The repository's `.gitignore` should exclude these by default while allowing explicitly curated example outputs to be added when appropriate.
