# SVEFusion VoD clean, faulty, reconstructed comparison

## Published SVEFusion preprocessing baseline

For a paper-aligned **clean** reference, use `prepare_vod_official.py` before
the fault comparison below. It links unmodified VoD LiDAR and seven-channel
five-frame radar into the `rlfusion_5f` layout described by SVEFusion/L4DR,
generates the original train and validation infos with point counts, and
builds the fusion ground-truth sampling database. The generated model YAML
preserves the upstream field-of-view, augmentation, voxelization, and VoD
evaluation settings. The generated test YAML switches only to the public
unlabeled test IDs. Use a separate directory from the custom detector export.

```bash
SVE=/path/to/SVEFusion
VOD=/path/to/view_of_delft_PUBLIC
PYTHON=/path/to/svefusion_environment/bin/python
OFFICIAL=/path/to/svefusion_official_vod

cd "$SVE"
"$PYTHON" tools/prepare_vod_official.py \
  --vod-public "$VOD" --data-root "$OFFICIAL" \
  --prepare-infos --prepare-database --workers 4

cd "$SVE/tools"
"$PYTHON" test.py \
  --cfg_file cfgs/VoD_models/SVEFusion_vod_official_clean.yaml \
  --ckpt /path/to/svefusion_vod.pth --batch_size 1 --workers 2
```

This clean baseline uses the published input processing; it is independent of
the custom LiDAR fault and reconstruction exports below. Those conditions are
outside the SVEFusion paper and must be regenerated from full scans to avoid
the forward-point filter used by the earlier exporter. Keep the detector
checkpoint fixed across conditions. Public test IDs have no public labels, so
the test config saves predictions without local AP.

To compare the same validation IDs with clean and faulty LiDAR under those
published transforms, use the full-scan fault artifacts. This creates separate
condition roots, preserving raw radar and calibration. It automatically limits
both roots to the same frames if the fault cache is incomplete.

```bash
CACHE=/path/to/vod_range5_full_cache
MATCHED=/path/to/svefusion_official_matched
"$PYTHON" "$SVE/tools/prepare_vod_official_faults.py" \
  --clean-root "$OFFICIAL" --fault-samples-root "$CACHE/samples" \
  --output-root "$MATCHED" --split val

cd "$SVE/tools"
for condition in clean faulty; do
  "$PYTHON" test.py \
    --cfg_file "cfgs/VoD_models/SVEFusion_vod_official_${condition}_val.yaml" \
    --ckpt /path/to/svefusion_vod.pth --batch_size 1 --workers 2 \
    --extra_tag upstream_processing --eval_tag "$condition"
done
```

Add `--reconstructed-export-root /path/to/lidar/reconstructed` if the previous
reconstruction export is available. When that export contains only forward
points, the preparation script restores unmodified rear LiDAR from the raw
scan, leaving SVEFusion's own camera FoV filter to choose model inputs. The
reconstruction itself remains a custom intervention. Use `--split test` for
matched unlabeled test predictions once test fault artifacts exist.

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
FAULT_PYTHON=/path/to/reconstruction_environment/bin/python
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

## Official test split inference

`test.py` with the validation configs above still reads the labeled `val` split.
Use `prepare_vod_test.py` to create separate configs for the official `test`
IDs. Public VoD test labels are unavailable, so this writes predictions but
does not produce test AP. The script leaves the validation configs and infos
in place.

```bash
cd "$SVE"
"$PYTHON" tools/prepare_vod_test.py \
  --export-root "$EXPORT" --vod-public "$VOD" --conditions clean --workers 4

cd "$SVE/tools"
"$PYTHON" test.py \
  --cfg_file cfgs/VoD_models/SVEFusion_vod_clean_test.yaml \
  --ckpt "$CKPT" --batch_size 1 --workers 2 \
  --extra_tag official_test --eval_tag clean --save_to_file
```

For a matched faulty comparison, first generate test faults with the same
fault plan and radar stack policy used for validation. The fault generator
resumes existing samples and stores them under `$CACHE/samples/test`.

```bash
cd "$REPO"
"$FAULT_PYTHON" -m scripts.create_vod_range_view_dataset \
  --vod-root "$VOD" --radar-cache-root "$CACHE/radar" \
  --output-root "$CACHE/samples" --split test

cd "$SVE"
"$PYTHON" tools/prepare_vod_test.py \
  --export-root "$EXPORT" --vod-public "$VOD" \
  --conditions clean faulty --fault-samples-root "$CACHE/samples" --workers 4

cd "$SVE/tools"
for condition in clean faulty; do
  "$PYTHON" test.py \
    --cfg_file "cfgs/VoD_models/SVEFusion_vod_${condition}_test.yaml" \
    --ckpt "$CKPT" --batch_size 1 --workers 2 \
    --extra_tag official_test_matched --eval_tag "$condition" --save_to_file
done
```

Fault generation may skip recording warm-up frames with fewer than five radar
scans. In that case, preparation selects the same subset of official test IDs
for both conditions. The saved `result.pkl` and KITTI-format text files can
be inspected or submitted to an official evaluator if one is available.

The raw VoD files are present in this Windows workspace under
`C:\Users\gianl\Desktop\Thesis\View-Of-Delft dataset\view_of_delft_PUBLIC`.
The complete range-view reconstruction export, a selected reconstruction
checkpoint, and a trained SVEFusion checkpoint are needed before numerical
scores can be produced.
