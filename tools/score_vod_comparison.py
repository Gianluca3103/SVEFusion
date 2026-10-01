"""Score three matched SVEFusion result.pkl files with KITTI 3D AP_R40."""

import argparse
import csv
import pickle
from pathlib import Path

from pcdet.datasets.vod.kitti_object_eval_python.eval import get_official_eval_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-infos", type=Path, required=True)
    for condition in ("clean", "faulty", "reconstructed"):
        parser.add_argument(f"--{condition}-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.clean_infos.open("rb") as handle:
        infos = pickle.load(handle)
    ids = [str(info["point_cloud"]["lidar_idx"]) for info in infos]
    ground_truth = [info["annos"] for info in infos]
    rows = []
    for condition in ("clean", "faulty", "reconstructed"):
        with getattr(args, f"{condition}_results").open("rb") as handle:
            predictions = pickle.load(handle)
        if [str(item["frame_id"]) for item in predictions] != ids:
            raise ValueError(f"{condition} prediction IDs or order differ from validation infos")
        _, scores = get_official_eval_result(
            ground_truth, predictions, ["Car", "Pedestrian", "Cyclist"])
        row = {"condition": condition, "frames": len(ids)}
        for name in ("Car", "Pedestrian", "Cyclist"):
            for difficulty in ("easy", "moderate", "hard"):
                key = f"{name}_3d/{difficulty}_R40"
                row[key] = float(scores[key])
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row["condition"], {name: row[f"{name}_3d/moderate_R40"]
                                 for name in ("Car", "Pedestrian", "Cyclist")})
    print(args.output)


if __name__ == "__main__":
    main()
