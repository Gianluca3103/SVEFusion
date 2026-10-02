"""Make matched VoD clean/faulty SVEFusion inputs from full-scan fault artifacts.

Requires a clean reference made by prepare_vod_official.py. The test or
validation condition directories keep the upstream SVEFusion YAML processing;
only the chosen LiDAR files differ. Reconstructed files are optional because
they must first be produced by the separate range-view model.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import yaml

from prepare_vod_official import link_directory, read_ids


def find_faults(root: Path, split: str, official: set[str]) -> dict[str, Path]:
    samples = {}
    for path in sorted((root / split).glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata_json"].item()))
        frame = str(meta.get("frame_id", ""))
        if (meta.get("dataset") != "View-of-Delft" or meta.get("split") != split
                or meta.get("range_view_full_scan") is not True
                or frame not in official or frame in samples):
            raise ValueError(f"Invalid or duplicate {split} fault sample: {path}")
        samples[frame] = path
    if not samples:
        raise FileNotFoundError(f"No full-scan fault artifacts in {root / split}")
    return samples


def write_cloud(path: Path, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
        raise ValueError(f"Expected finite LiDAR [N,4]: {path}")
    desired = np.ascontiguousarray(points, dtype="<f4").reshape(-1)
    if path.exists():
        if not np.array_equal(np.fromfile(path, dtype="<f4"), desired):
            raise FileExistsError(f"Existing LiDAR differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    desired.tofile(path)


def prepare_layout(clean_root: Path, root: Path, split: str,
                   ids: list[str], condition: str) -> None:
    partition = "testing" if split == "test" else "training"
    for part in ("training", "testing"):
        for name in ("calib", "image_2", "lidar_calib", "radar_5f", "radar_calib"):
            link_directory(clean_root / part / name, root / part / name)
        if part == "training":
            link_directory(clean_root / part / "label_2", root / part / "label_2")
        if part != partition or condition == "clean":
            link_directory(clean_root / part / "lidar", root / part / "lidar")
    if condition != "clean" and split == "val":
        lidar = root / "training" / "lidar"
        lidar.mkdir(parents=True, exist_ok=True)
        for frame in read_ids(clean_root / "ImageSets" / "train.txt"):
            source = clean_root / "training" / "lidar" / f"{frame}.bin"
            destination = lidar / source.name
            if not destination.exists():
                destination.symlink_to(source.resolve())
            elif destination.resolve() != source.resolve():
                raise FileExistsError(f"Training LiDAR differs: {destination}")

    image_sets = root / "ImageSets"
    image_sets.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        source = read_ids(clean_root / "ImageSets" / f"{name}.txt")
        chosen = ids if name == split else source
        (image_sets / f"{name}.txt").write_text(
            "\n".join(chosen) + "\n", encoding="utf-8")
    train_ids = read_ids(image_sets / "train.txt")
    val_ids = read_ids(image_sets / "val.txt")
    (image_sets / "trainval.txt").write_text(
        "\n".join(train_ids + val_ids) + "\n", encoding="utf-8")
    if (clean_root / "gt_database").is_dir():
        link_directory(clean_root / "gt_database", root / "gt_database")
    for name in ("vod_dbinfos_train.pkl", "vod_infos_train.pkl"):
        source = clean_root / name
        if source.is_file():
            destination = root / name
            if not destination.exists():
                destination.symlink_to(source.resolve())
            elif destination.resolve() != source.resolve():
                raise FileExistsError(f"Existing info differs: {destination}")
    for name in ("val", "test"):
        source = clean_root / f"vod_infos_{name}.pkl"
        with source.open("rb") as handle:
            infos = pickle.load(handle)
        wanted = set(read_ids(image_sets / f"{name}.txt"))
        infos = [item for item in infos
                 if str(item["point_cloud"]["lidar_idx"]) in wanted]
        if len(infos) != len(wanted):
            raise ValueError(f"Incomplete {name} infos for {condition}")
        with (root / f"vod_infos_{name}.pkl").open("wb") as handle:
            pickle.dump(infos, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_model_config(sve_root: Path, root: Path, split: str,
                       condition: str) -> Path:
    cfgs = sve_root / "tools" / "cfgs"
    with (cfgs / "dataset_configs" / "Vod_fusion.yaml").open() as handle:
        dataset = yaml.safe_load(handle)
    with (cfgs / "VoD_models" / "SVEFusion.yaml").open() as handle:
        model = yaml.safe_load(handle)
    dataset["DATA_PATH"] = str(root.resolve())
    name = f"vod_sve_official_{condition}_{split}.yaml"
    (cfgs / "dataset_configs" / name).write_text(
        yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    model["DATA_CONFIG"]["_BASE_CONFIG_"] = f"cfgs/dataset_configs/{name}"
    if split == "test":
        model["DATA_CONFIG"]["DATA_SPLIT"] = {"train": "train", "test": "test"}
        model["DATA_CONFIG"]["INFO_PATH"] = {
            "train": ["kitti_infos_train.pkl", "vod_infos_train.pkl"],
            "test": ["vod_infos_test.pkl"],
        }
    path = cfgs / "VoD_models" / f"SVEFusion_vod_official_{condition}_{split}.yaml"
    path.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sve-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--fault-samples-root", type=Path, required=True)
    parser.add_argument("--reconstructed-export-root", type=Path,
                        help="Optional prior lidar/reconstructed export; rear LiDAR is restored if forward-only")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    args = parser.parse_args()
    clean_root = args.clean_root.resolve()
    official_ids = read_ids(clean_root / "ImageSets" / f"{args.split}.txt")
    faults = find_faults(args.fault_samples_root, args.split, set(official_ids))
    ids = [frame for frame in official_ids if frame in faults]
    partition = "testing" if args.split == "test" else "training"
    print(f"Matched {len(ids)}/{len(official_ids)} {args.split} IDs", flush=True)
    reconstruction = args.reconstructed_export_root
    forward_only = False
    if reconstruction is not None:
        manifest = json.loads((reconstruction / "export_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("condition") != "reconstructed" or manifest.get("mode") != "lidar":
            raise ValueError("Expected a reconstructed LiDAR export")
        forward_only = manifest.get("forward_only")
        if not isinstance(forward_only, bool):
            raise ValueError("Reconstruction export lacks a valid forward-only policy")
        for frame in ids:
            path = reconstruction / partition / "velodyne" / f"{frame}.bin"
            if not path.is_file():
                raise FileNotFoundError(path)

    conditions = ("clean", "faulty", "reconstructed") if reconstruction else ("clean", "faulty")
    for condition in conditions:
        root = args.output_root / args.split / condition
        root.mkdir(parents=True, exist_ok=True)
        prepare_layout(clean_root, root, args.split, ids, condition)
        if condition == "faulty":
            lidar = root / partition / "lidar"
            for index, frame in enumerate(ids, 1):
                with np.load(faults[frame], allow_pickle=False) as archive:
                    points = archive["faulty_lidar_points"]
                write_cloud(lidar / f"{frame}.bin", points)
                if index % 250 == 0 or index == len(ids):
                    print(f"Faulty LiDAR: {index}/{len(ids)}", flush=True)
        elif condition == "reconstructed":
            lidar = root / partition / "lidar"
            for index, frame in enumerate(ids, 1):
                path = reconstruction / partition / "velodyne" / f"{frame}.bin"
                points = np.fromfile(path, dtype="<f4").reshape(-1, 4)
                if forward_only:
                    sentinel = np.array([0.01, 0.0, -2.9, 0.0], dtype=np.float32)
                    if len(points) == 1 and np.array_equal(points[0], sentinel):
                        points = points[:0]
                    raw = np.fromfile(clean_root / partition / "lidar" / f"{frame}.bin",
                                      dtype="<f4").reshape(-1, 4)
                    points = np.concatenate((points, raw[raw[:, 0] < 0]), axis=0)
                write_cloud(lidar / f"{frame}.bin", points)
                if index % 250 == 0 or index == len(ids):
                    print(f"Reconstructed LiDAR: {index}/{len(ids)}", flush=True)
        config = write_model_config(args.sve_root, root, args.split, condition)
        (root / "preparation_manifest.json").write_text(json.dumps({
            "condition": condition, "split": args.split, "frames": len(ids),
            "lidar": {
                "clean": "unmodified clean",
                "faulty": "full-scan injected fault",
                "reconstructed": "range-view reconstruction with raw rear LiDAR restored",
            }[condition],
            "radar": "unmodified seven-channel five-frame VoD",
            "config": str(config),
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Config: {config}", flush=True)


if __name__ == "__main__":
    main()
