# SVEFusion VoD clean, faulty, reconstructed comparison

Train **one SVEFusion model on clean LiDAR plus five-frame radar**, then evaluate
that same checkpoint on the same VoD validation IDs with clean, faulty, and
reconstructed LiDAR. Radar is identical in all three evaluations. The output
CSV contains KITTI 3D AP_R40 for Car, Pedestrian, and Cyclist at each
difficulty. This is a native SVEFusion two-branch fusion experiment: LiDAR is
four-channel XYZI; radar remains seven-channel XYZ, RCS, radial velocity,
compensated radial velocity, and time. The loader transforms radar points into
the LiDAR coordinate frame using the two VoD calibration files.

The existing `Sensor-Fusion_Final_Model_Repo/scripts/export_vod_pvrcnn.py`
generates matched LiDAR conditions. Use its **lidar-only** output; do not pass
`--with-radar`, which concatenates radar into a four-channel cloud for a
different detector. That exporter verifies official train/val membership,
normalizes `bicycle` labels to `Cyclist`, uses the same labels and IDs in all
conditions, trains on clean scans in every export, and changes LiDAR only on
validation. Select a range-view reconstruction checkpoint before export.

## GPU-machine workflow

These commands assume a Linux CUDA installation of this SVEFusion checkout and
the reconstruction repository. Replace the paths with their locations on the
GPU machine. `VOD` is the directory containing `lidar`, `radar`, and
`radar_5frames`; `CACHE` contains range-view `samples/{train,val}` and `radar`.

```bash
SVE=/path/to/SVEFusion-main
REPO=/path/to/Sensor-Fusion_Final_Model_Repo
VOD=/path/to/view_of_delft_PUBLIC
CACHE=/path/to/vod_range5_full_cache
RECON=/path/to/range_view_checkpoint.pt
EXPORT=/path/to/sve_vod_comparison
PYTHON=/path/to/cuda_environment/bin/python
export PYTHONPATH="$REPO:$SVE${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO"
"$PYTHON" -m scripts.export_vod_pvrcnn \
  --vod-root "$VOD" --samples-root "$CACHE/samples" \
  --radar-root "$CACHE/radar" --checkpoint "$RECON" \
  --output-root "$EXPORT" --device cpu

cd "$SVE"
"$PYTHON" tools/prepare_vod_comparison.py \
  --export-root "$EXPORT" --vod-public "$VOD" --prepare-infos --workers 4

cd "$SVE/tools"
"$PYTHON" train.py --cfg_file cfgs/VoD_models/SVEFusion_vod_clean.yaml \
  --batch_size 4 --workers 4 --extra_tag vod_clean_trained \
  --val_interval 5
```

The preparation command writes three dataset/model configs and shared
validation infos. It disables ground-truth database sampling, so no fusion GT
database is required. It checks that all validation files and calibration
files exist and that condition IDs match. Only the `clean` config is for
training. With `--val_interval 5`, validation runs on the same GPU after every
fifth epoch and at the final epoch. The training log and TensorBoard report
Car, Pedestrian, Cyclist, and mean moderate 3D AP_R40. Set the interval to 1
for every epoch. The AP_R40 evaluator uses the bundled CPU rotated-IoU
implementation for compatibility with current Numba releases, so validation
adds time to those epochs. Use a smaller batch if memory requires it. If `--device cpu` makes
the reconstruction export too slow, choose `cuda` when the GPU is free.

If only the clean LiDAR export exists, add `--conditions clean` to the
preparation command and train with the clean config. Once faulty and
reconstructed validation scans have been exported, rerun preparation without
that option before the three-condition evaluation.

Find the clean training checkpoint under the SVEFusion `output/VoD_models/`
directory, then run:

```bash
CKPT=/path/to/checkpoint_epoch_80.pth
bash "$SVE/tools/scripts/eval_vod_comparison.sh" \
  "$SVE" "$EXPORT" "$CKPT" "$PYTHON"
```

Each test run writes `result.pkl` beneath its condition's `output/VoD_models/`
evaluation directory. Pass the three exact files to the scorer:

```bash
cd "$SVE"
"$PYTHON" tools/score_vod_comparison.py \
  --clean-infos "$EXPORT/lidar/clean/vod_infos_val.pkl" \
  --clean-results /path/to/clean/result.pkl \
  --faulty-results /path/to/faulty/result.pkl \
  --reconstructed-results /path/to/reconstructed/result.pkl \
  --output "$EXPORT/svefusion_3d_ap_r40.csv"
```

The scorer refuses predictions with different validation IDs or ordering. It
uses the same evaluation implementation and IoU thresholds for all conditions.
For a quick pipeline check, append `--limit 10` to the export command; those
partial-set AP values are not full validation results.

The raw VoD files are present in this Windows workspace under
`C:\Users\gianl\Desktop\Thesis\View-Of-Delft dataset\view_of_delft_PUBLIC`.
The complete range-view reconstruction export, a selected reconstruction
checkpoint, and a trained SVEFusion checkpoint are needed before numerical
scores can be produced.
