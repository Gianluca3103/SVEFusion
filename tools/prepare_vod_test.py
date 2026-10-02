"""Prepare the official, potentially unlabeled VoD test split for SVEFusion.

Clean scans come from the public VoD release. Faulty scans come from the same
range-view fault artifacts used for the validation comparison. Both conditions
use identical frame IDs whenever faulty scans are requested.
"""

import argparse
import contextlib
import copy
import json
import os
import pickle
from pathlib import Path

import numpy as np
import yaml


def link_directory(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f"Expected {destination} to point to {source}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=True)


def write_bin(destination: Path, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
        raise ValueError(f"Expected finite [N,4] LiDAR points for {destination}")
    values = np.ascontiguousarray(points, dtype="<f4")
    if destination.exists():
        old = np.fromfile(destination, dtype="<f4")
        if not np.array_equal(old, values.reshape(-1)):
            raise FileExistsError(f"Existing test cloud differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    values.tofile(destination)


def fault_index(samples_root: Path, official_ids: set[str]) -> dict[str, Path]:
    files = sorted((samples_root / "test").glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No faulty test artifacts in {samples_root / 'test'}")
    indexed = {}
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata_json"].item()))
        if meta.get("dataset") != "View-of-Delft" or meta.get("split") != "test" \
                or meta.get("range_view_full_scan") is not True:
            raise ValueError(f"Unexpected fault sample metadata: {path}")
        frame_id = str(meta["frame_id"])
        if frame_id not in official_ids or frame_id in indexed:
            raise ValueError(f"Unknown or duplicate test frame {frame_id}: {path}")
        indexed[frame_id] = path
    return indexed


def configure(sve_root: Path, export_root: Path, vod_public: Path,
              conditions: tuple[str, ...]) -> None:
    cfgs = sve_root / "tools" / "cfgs"
    with (cfgs / "dataset_configs" / "Vod_fusion.yaml").open() as handle:
        dataset_base = yaml.safe_load(handle)
    with (cfgs / "VoD_models" / "SVEFusion.yaml").open() as handle:
        model_base = yaml.safe_load(handle)
    radar_root = (vod_public / "radar_5frames" / "testing" / "velodyne").resolve()
    radar_calib = (vod_public / "radar" / "testing" / "calib").resolve()
    for path in (radar_root, radar_calib):
        if not path.is_dir():
            raise FileNotFoundError(path)
    for condition in conditions:
        root = (export_root / "lidar" / condition).resolve()
        dataset = copy.deepcopy(dataset_base)
        dataset.update({
            "DATA_PATH": str(root),
            "DATA_SPLIT": {"train": "train", "test": "test"},
            "INFO_PATH": {"train": ["vod_infos_train.pkl"],
                          "test": ["vod_infos_test.pkl"]},
            "LIDAR_POINTS_ROOT": str(root / "testing" / "velodyne"),
            "LIDAR_CALIB_ROOT": str(root / "testing" / "calib"),
            "RADAR_POINTS_ROOT": str(radar_root),
            "RADAR_CALIB_ROOT": str(radar_calib),
            "FOV_POINTS_ONLY": True,
            "VOD_EVA": False,
        })
        dataset["DATA_AUGMENTOR"]["DISABLE_AUG_LIST"] = ["placeholder", "gt_sampling"]
        dataset_name = f"vod_sve_{condition}_test.yaml"
        dataset_path = cfgs / "dataset_configs" / dataset_name
        dataset_path.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")

        model = copy.deepcopy(model_base)
        model["DATA_CONFIG"]["_BASE_CONFIG_"] = f"cfgs/dataset_configs/{dataset_name}"
        model["DATA_CONFIG"]["FOV_POINTS_ONLY"] = True
        model["DATA_CONFIG"]["VOD_EVA"] = False
        model["DATA_CONFIG"]["DATA_AUGMENTOR"]["DISABLE_AUG_LIST"] = ["placeholder", "gt_sampling"]
        model_path = cfgs / "VoD_models" / f"SVEFusion_vod_{condition}_test.yaml"
        model_path.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
        print(f"Config: {model_path}")


def prepare_infos(export_root: Path, sve_root: Path, workers: int,
                  conditions: tuple[str, ...], ids: list[str]) -> None:
    from easydict import EasyDict
    from pcdet.datasets.vod.vod_dataset import VodDataset

    clean = export_root / "lidar" / "clean"
    cfg = sve_root / "tools" / "cfgs" / "dataset_configs" / "vod_sve_clean_test.yaml"
    dataset_cfg = EasyDict(yaml.safe_load(cfg.read_text(encoding="utf-8")))
    dataset = VodDataset(dataset_cfg, ["Car", "Pedestrian", "Cyclist"],
                         training=False, root_path=clean)
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        infos = dataset.get_infos(num_workers=workers, has_label=False,
                                  count_inside_pts=False)
    if [str(info["point_cloud"]["lidar_idx"]) for info in infos] != ids:
        raise ValueError("Generated test info IDs differ from the selected frames")
    for condition in conditions:
        destination = export_root / "lidar" / condition / "vod_infos_test.pkl"
        with destination.open("wb") as handle:
            pickle.dump(infos, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Infos: {destination} ({len(infos)} unlabeled frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sve-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--vod-public", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=("clean", "faulty"),
                        default=("clean",))
    parser.add_argument("--fault-samples-root", type=Path,
                        help="Required for faulty inference; expects test/*.npz")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    conditions = tuple(dict.fromkeys(args.conditions))
    if "clean" not in conditions:
        parser.error("Include clean so both conditions share test frame IDs")
    if "faulty" in conditions and args.fault_samples_root is None:
        parser.error("--fault-samples-root is required for faulty test inference")

    public = args.vod_public.resolve()
    source = public / "lidar" / "testing"
    official = (public / "lidar" / "ImageSets" / "test.txt").read_text().split()
    if not official or len(official) != len(set(official)):
        raise ValueError("The official VoD test IDs are empty or duplicated")
    samples = fault_index(args.fault_samples_root, set(official)) if "faulty" in conditions else {}
    ids = [frame for frame in official if not samples or frame in samples]
    print(f"Selected {len(ids)}/{len(official)} official test frames")

    for condition in conditions:
        root = args.export_root / "lidar" / condition
        manifest = root / "export_manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("condition") != condition or payload.get("mode") != "lidar":
            raise ValueError(f"Unexpected export manifest: {manifest}")
        forward_only = payload.get("forward_only", True)
        if not isinstance(forward_only, bool):
            raise ValueError(f"Invalid forward-only policy in {manifest}")
        link_directory(source / "calib", root / "testing" / "calib")
        link_directory(source / "image_2", root / "testing" / "image_2")
        for index, frame in enumerate(ids, 1):
            raw = source / "velodyne" / f"{frame}.bin"
            if not raw.is_file() or not (source / "calib" / f"{frame}.txt").is_file():
                raise FileNotFoundError(f"Missing test LiDAR or calibration for {frame}")
            if not any((source / "image_2" / f"{frame}{ext}").is_file()
                       for ext in (".jpg", ".png")):
                raise FileNotFoundError(f"Missing test image for {frame}")
            if condition == "clean":
                points = np.fromfile(raw, dtype="<f4").reshape(-1, 4)
            else:
                with np.load(samples[frame], allow_pickle=False) as archive:
                    points = np.asarray(archive["faulty_lidar_points"], dtype=np.float32)
            if forward_only:
                points = points[points[:, 0] >= 0]
            write_bin(root / "testing" / "velodyne" / f"{frame}.bin", points)
            if index % 250 == 0 or index == len(ids):
                print(f"{condition}: {index}/{len(ids)} test scans", flush=True)
        image_sets = root / "ImageSets"
        image_sets.mkdir(parents=True, exist_ok=True)
        (image_sets / "test.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    configure(args.sve_root, args.export_root, public, conditions)
    prepare_infos(args.export_root, args.sve_root, args.workers, conditions, ids)


if __name__ == "__main__":
    main()
