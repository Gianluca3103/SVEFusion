"""Configure SVEFusion for the matched range-view VoD detector export.

First run Sensor-Fusion_Final_Model_Repo/scripts/export_vod_pvrcnn.py without
--with-radar. Its lidar/{clean,faulty,reconstructed} directories provide the
four-channel LiDAR input, labels, calibration, and fixed split IDs. Radar is
read separately from the original seven-channel VoD radar_5frames files.
"""

import argparse
import copy
import json
import pickle
from pathlib import Path

import yaml


CONDITIONS = ("clean", "faulty", "reconstructed")


def configure(sve_root: Path, export_root: Path, vod_public: Path) -> list[Path]:
    cfgs = sve_root / "tools" / "cfgs"
    with (cfgs / "dataset_configs" / "Vod_fusion.yaml").open() as handle:
        dataset_base = yaml.safe_load(handle)
    with (cfgs / "VoD_models" / "SVEFusion.yaml").open() as handle:
        model_base = yaml.safe_load(handle)
    radar_root = (vod_public / "radar_5frames" / "training" / "velodyne").resolve()
    radar_calib = (vod_public / "radar" / "training" / "calib").resolve()
    for path in (radar_root, radar_calib):
        if not path.is_dir():
            raise FileNotFoundError(path)
    expected_splits = None
    outputs = []
    for condition in CONDITIONS:
        root = (export_root / "lidar" / condition).resolve()
        manifest = root / "export_manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("condition") != condition or payload.get("mode") != "lidar":
            raise ValueError(f"Unexpected export manifest: {manifest}")
        splits = {name: (root / "ImageSets" / f"{name}.txt").read_text().split()
                  for name in ("train", "val")}
        if expected_splits is None:
            expected_splits = splits
        elif splits != expected_splits:
            raise ValueError(f"Mismatched split IDs in {root}")
        if set(splits["train"]) & set(splits["val"]):
            raise ValueError("Train and validation IDs overlap")
        lidar_root = root / "training" / "velodyne"
        for frame in splits["val"]:
            for path in (lidar_root / f"{frame}.bin", radar_root / f"{frame}.bin",
                         radar_calib / f"{frame}.txt"):
                if not path.is_file():
                    raise FileNotFoundError(path)
        dataset = copy.deepcopy(dataset_base)
        dataset.update({
            "DATA_PATH": str(root),
            "LIDAR_POINTS_ROOT": str(lidar_root),
            "LIDAR_CALIB_ROOT": str(root / "training" / "calib"),
            "RADAR_POINTS_ROOT": str(radar_root),
            "RADAR_CALIB_ROOT": str(radar_calib),
            "INFO_PATH": {"train": ["vod_infos_train.pkl"],
                          "test": ["vod_infos_val.pkl"]},
            "FOV_POINTS_ONLY": False,
        })
        dataset["DATA_AUGMENTOR"]["DISABLE_AUG_LIST"] = ["placeholder", "gt_sampling"]
        dataset_path = cfgs / "dataset_configs" / f"vod_sve_{condition}.yaml"
        model = copy.deepcopy(model_base)
        model["DATA_CONFIG"]["_BASE_CONFIG_"] = f"cfgs/dataset_configs/{dataset_path.name}"
        model["DATA_CONFIG"]["FOV_POINTS_ONLY"] = False
        model["DATA_CONFIG"]["DATA_AUGMENTOR"]["DISABLE_AUG_LIST"] = ["placeholder", "gt_sampling"]
        model_path = cfgs / "VoD_models" / f"SVEFusion_vod_{condition}.yaml"
        for path, document in ((dataset_path, dataset), (model_path, model)):
            path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        outputs.append(model_path)
    return outputs


def prepare_infos(export_root: Path, sve_root: Path, workers: int) -> None:
    from easydict import EasyDict
    from pcdet.datasets.vod.vod_dataset import VodDataset

    clean = (export_root / "lidar" / "clean").resolve()
    config = sve_root / "tools" / "cfgs" / "dataset_configs" / "vod_sve_clean.yaml"
    dataset_cfg = EasyDict(yaml.safe_load(config.read_text(encoding="utf-8")))
    dataset = VodDataset(dataset_cfg, ["Car", "Pedestrian", "Cyclist"],
                         training=False, root_path=clean)
    for split in ("train", "val"):
        dataset.set_split(split)
        infos = dataset.get_infos(num_workers=workers, has_label=True,
                                  count_inside_pts=False)
        ids = (clean / "ImageSets" / f"{split}.txt").read_text().split()
        if [str(info["point_cloud"]["lidar_idx"]) for info in infos] != ids:
            raise ValueError(f"Unexpected {split} info IDs")
        destinations = CONDITIONS if split == "val" else ("clean",)
        for condition in destinations:
            target = export_root / "lidar" / condition / f"vod_infos_{split}.pkl"
            with target.open("wb") as handle:
                pickle.dump(infos, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"{target}: {len(infos)} frames")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sve-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--vod-public", type=Path, required=True)
    parser.add_argument("--prepare-infos", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    for path in configure(args.sve_root, args.export_root, args.vod_public):
        print(path)
    if args.prepare_infos:
        prepare_infos(args.export_root, args.sve_root, args.workers)


if __name__ == "__main__":
    main()
