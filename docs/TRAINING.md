# Training QHiFi

Training is launched by `main.py`, which constructs `AugmentedDataset` instances and calls `train.py`.

## Data-list format

Each line in the training, validation, and test list files must contain three whitespace-separated relative paths in this order:

```text
path/to/WF_image.tif path/to/SIM_image.tif path/to/PSF.tif
```

For example:

```text
Results80/WF/001.tif Results80/SIM/001.tif PSF/psf.tif
Results85/WF/001.tif Results85/SIM/001.tif PSF/psf.tif
```

The `--datapath` value is prepended to each listed path. The three list files may reference different samples. Do not change an existing train/validation/test split when reproducing the reported experiment.

## Paper configuration

The confirmed final loss settings and random seed are:

```text
weight = 0.5
loss_corrected_l1 = 0.1
loss_corrected_ssim = 0.1
seed = 42
```

The three loss weights are configured in `main.py`, not exposed as command-line parameter-sweep options in the public training instructions. The ablation flags reported in the paper are retained. For a checkpoint trained with an ablation, make sure the corresponding switches are used consistently when constructing the model for evaluation.

## Example command

Run from `QHiFi-SIM main/`. Edit the data paths to match the local dataset layout first:

```bash
python main.py \
  --train_or_test train \
  --model_name QHiFi \
  --datapath ../dataset_QHiFi/MT/ \
  --datalist ../dataset_QHiFi/MT/train_list.txt \
  --evallist ../dataset_QHiFi/MT/eval_list.txt \
  --testlist ../dataset_QHiFi/MT/test_list.txt \
  --nEpochs 300 \
  --batch_size 16 \
  --lr 1e-4 \
  --patch_size 256 \
  --validate_interval 500 \
  --sample_interval 1000 \
  --validate_num 4000 \
  --sample_num 3 \
  --model_save_frequency 1 \
  --experiment_name paper_final \
  --seed 42
```

This command documents the settings visible in the supplied training launcher except for the final loss weights, which are set in `main.py`. Check that the evaluation-set size supports the requested validation sample count and that the output/checkpoint paths are writable. Use the actual epoch count and intervals that correspond to the experiment you intend to reproduce.

## Outputs

Training writes model checkpoints under the configured checkpoint directory, logs training/validation metrics, and writes TensorBoard events under `runs/`. These are generated outputs and should not normally be committed to Git.

## Reproducibility note

Random seeds improve repeatability but do not guarantee bitwise-identical weights across all hardware, CUDA, PyTorch, and library versions. The documented configuration should be used with the source revision and environment corresponding to the paper. Do not silently substitute a modified training-mode implementation for the code that generated the published checkpoint.
