"""Build the L4DR/SVEFusion VoD layout from unmodified public sensor files.

Only DATA_PATH, the base-config reference, and the test split are changed in
the upstream YAML configs.
The default model config evaluates the labeled validation set. A second config
uses the unlabeled official test set. No LiDAR clipping or radar re-encoding is
performed here; SVEFusion applies its own FoV filtering and voxelization.
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


def read_ids(path: Path) -> list[str]:
    ids = path.read_text(encoding="utf-8").split()
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Empty or duplicated split IDs: {path}")
    return ids


def layout(public: Path, destination: Path) -> dict[str, list[str]]:
    source_ids = public / "lidar" / "ImageSets"
    splits = {name: read_ids(source_ids / f"{name}.txt")
              for name in ("train", "val", "test")}
    if any(set(splits[a]) & set(splits[b])
           for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Official VoD split IDs overlap")

    for partition in ("training", "testing"):
        lidar = public / "lidar" / partition
        radar = public / "radar" / partition
        radar_five = public / "radar_5frames" / partition
        targets = {
            "calib": lidar / "calib",
            "image_2": lidar / "image_2",
            "lidar": lidar / "velodyne",
            "lidar_calib": lidar / "calib",
            "radar_5f": radar_five / "velodyne",
            "radar_calib": radar / "calib",
        }
        if partition == "training":
            targets["label_2"] = lidar / "label_2"
        for name, source in targets.items():
            link_directory(source, destination / partition / name)
        if (radar / "velodyne").is_dir():
            link_directory(radar / "velodyne", destination / partition / "radar")
        pose = lidar / "pose"
        if pose.is_dir():
            link_directory(pose, destination / partition / "pose")

    image_sets = destination / "ImageSets"
    image_sets.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (image_sets / f"{name}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    (image_sets / "trainval.txt").write_text(
        "\n".join(splits["train"] + splits["val"]) + "\n", encoding="utf-8")

    for split, ids in splits.items():
        partition = "testing" if split == "test" else "training"
        for frame in ids:
            required = [
                destination / partition / "lidar" / f"{frame}.bin",
                destination / partition / "radar_5f" / f"{frame}.bin",
                destination / partition / "lidar_calib" / f"{frame}.txt",
                destination / partition / "radar_calib" / f"{frame}.txt",
            ]
            if split != "test":
                required.append(destination / partition / "label_2" / f"{frame}.txt")
            if not any((destination / partition / "image_2" / f"{frame}{ext}").is_file()
                       for ext in (".jpg", ".png")):
                raise FileNotFoundError(f"Missing {split} image for {frame}")
            for path in required:
                if not path.is_file():
                    raise FileNotFoundError(path)
            lidar_path, radar_path = required[:2]
            if lidar_path.stat().st_size % (4 * 4):
                raise ValueError(f"LiDAR is not four-channel float32: {lidar_path}")
            if radar_path.stat().st_size % (7 * 4):
                raise ValueError(f"Radar is not seven-channel float32: {radar_path}")
        print(f"Verified {len(ids)} {split} LiDAR/radar pairs", flush=True)
        first_radar = destination / partition / "radar_5f" / f"{ids[0]}.bin"
        radar_points = np.fromfile(first_radar, dtype="<f4").reshape(-1, 7)
        if len(radar_points) and (not np.isfinite(radar_points).all()
                                  or not np.isin(radar_points[:, 6], [-4, -3, -2, -1, 0]).all()):
            raise ValueError(f"Unexpected radar values or time indices: {first_radar}")
    return splits


def write_configs(sve_root: Path, destination: Path) -> tuple[Path, Path]:
    cfgs = sve_root / "tools" / "cfgs"
    with (cfgs / "dataset_configs" / "Vod_fusion.yaml").open() as handle:
        dataset = yaml.safe_load(handle)
    with (cfgs / "VoD_models" / "SVEFusion.yaml").open() as handle:
        model = yaml.safe_load(handle)
    dataset["DATA_PATH"] = str(destination.resolve())
    dataset_name = "vod_sve_official_clean.yaml"
    dataset_path = cfgs / "dataset_configs" / dataset_name
    dataset_path.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")

    model["DATA_CONFIG"]["_BASE_CONFIG_"] = f"cfgs/dataset_configs/{dataset_name}"
    val_path = cfgs / "VoD_models" / "SVEFusion_vod_official_clean.yaml"
    val_path.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")

    test_model = copy.deepcopy(model)
    test_model["DATA_CONFIG"]["DATA_SPLIT"] = {"train": "train", "test": "test"}
    test_model["DATA_CONFIG"]["INFO_PATH"] = {
        "train": ["kitti_infos_train.pkl", "vod_infos_train.pkl"],
        "test": ["vod_infos_test.pkl"],
    }
    test_path = cfgs / "VoD_models" / "SVEFusion_vod_official_clean_test.yaml"
    test_path.write_text(yaml.safe_dump(test_model, sort_keys=False), encoding="utf-8")
    print(f"Validation config: {val_path}")
    print(f"Unlabeled test config: {test_path}")
    return dataset_path, val_path


def prepare_infos(dataset_path: Path, destination: Path, workers: int,
                  splits: dict[str, list[str]], make_database: bool) -> None:
    from easydict import EasyDict
    from pcdet.datasets.vod.vod_dataset import VodDataset

    cfg = EasyDict(yaml.safe_load(dataset_path.read_text(encoding="utf-8")))
    dataset = VodDataset(cfg, ["Car", "Pedestrian", "Cyclist"],
                         training=False, root_path=destination)
    log_path = destination / "preparation.log"
    labeled_infos = {}
    for split in ("train", "val", "test"):
        dataset.set_split(split)
        print(f"Preparing {split} infos ({len(splits[split])} frames); details in {log_path}",
              flush=True)
        with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            infos = dataset.get_infos(num_workers=workers, has_label=split != "test",
                                      count_inside_pts=split != "test")
        if [str(item["point_cloud"]["lidar_idx"]) for item in infos] != splits[split]:
            raise ValueError(f"Generated {split} infos have incorrect frame IDs")
        path = destination / f"vod_infos_{split}.pkl"
        with path.open("wb") as handle:
            pickle.dump(infos, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if split != "test":
            labeled_infos[split] = infos
        print(f"Infos: {path} ({len(infos)} frames)", flush=True)
    with (destination / "vod_infos_trainval.pkl").open("wb") as handle:
        pickle.dump(labeled_infos["train"] + labeled_infos["val"], handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    if make_database:
        dataset.set_split("train")
        print(f"Preparing train ground-truth database; details in {log_path}", flush=True)
        with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            dataset.create_groundtruth_database(destination / "vod_infos_train.pkl",
                                                split="train")
        print(f"Ground-truth database: {destination / 'vod_dbinfos_train.pkl'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sve-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--vod-public", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prepare-infos", action="store_true")
    parser.add_argument("--prepare-database", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.prepare_database and not args.prepare_infos:
        parser.error("--prepare-database requires --prepare-infos")
    destination = args.data_root.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    splits = layout(args.vod_public.resolve(), destination)
    dataset_path, _ = write_configs(args.sve_root, destination)
    if args.prepare_infos:
        prepare_infos(dataset_path, destination, args.workers, splits,
                      args.prepare_database)
    summary = {"data_root": str(destination), "splits": {
        name: len(ids) for name, ids in splits.items()},
        "sensor_data": "unmodified VoD", "radar": "radar_5frames, seven channels"}
    (destination / "preparation_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
