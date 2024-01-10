import argparse
from pathlib import Path

import cv2
import numpy as np

from utils.annotation import ImageTextAnnotation
from utils.mot import BoundingBox, DetectionLabels, frame_from_video
from utils.util import box_iou


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording-dir", type=Path, default="data/recording", help="Path to the recording directory.")
    parser.add_argument(
        "--gold-annotation-dir",
        "-a",
        type=Path,
        default="data/image_text_annotation",
        help="Path to the gold image text annotation file.",
    )
    parser.add_argument(
        "--prediction-mot-dir", type=Path, default="result/mot", help="Path to the prediction directory."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output video directory.")
    parser.add_argument("--show", action="store_true", help="Show video while processing.")
    parser.add_argument("--scenario-ids", "--ids", type=str, nargs="*", help="List of scenario ids.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1337)
    for scenario_id in args.scenario_ids:
        image_text_annotation = ImageTextAnnotation.from_json(
            args.gold_annotation_dir.joinpath(f"{scenario_id}.json").read_text()
        )
        pred_mot = DetectionLabels.from_json(args.prediction_mot_dir.joinpath(f"{scenario_id}.json").read_text())
        class_names = pred_mot.class_names
        colors: np.ndarray = (rng.random(len(class_names) * 3) * 255).astype(np.uint8).reshape(-1, 3)  # (names, 3)

        instance_ids = set()
        for idx, image_annotation in enumerate(image_text_annotation.images):
            frame_prediction = pred_mot.frames[idx * 30]
            for bounding_box in frame_prediction.bounding_boxes:
                if any(box_iou(bb.rect, bounding_box.rect) >= 0.5 for bb in image_annotation.bounding_boxes):
                    instance_ids.add(bounding_box.instance_id)

        video = cv2.VideoCapture(str(args.recording_dir / scenario_id / "fp_video.mp4"))

        tagged_images = []
        frame: np.ndarray  # (h, w, 3), BGR
        for frame, frame_prediction in zip(frame_from_video(video), pred_mot.frames):
            bounding_box: BoundingBox
            for bounding_box in frame_prediction.bounding_boxes:
                if bounding_box.instance_id in instance_ids:
                    box_color = (150, 50, 255)  # BGR
                else:
                    box_color = (255, 255, 255)
                rect = bounding_box.rect
                color: list[int] = colors[class_names.index(bounding_box.class_name)].tolist()
                cv2.rectangle(frame, pt1=(rect.x1, rect.y1), pt2=(rect.x2, rect.y2), color=box_color, thickness=5)
                cv2.putText(
                    frame,
                    f"{bounding_box.class_name}_{bounding_box.instance_id}",
                    (rect.x1, rect.y1),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=5,
                    thickness=5,
                    color=color,
                )
            if args.show:
                cv2.imshow("img", frame)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
            tagged_images.append(frame)

        fourcc: int = cv2.VideoWriter_fourcc(*"xvid")
        w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(output_dir / scenario_id), fourcc, 30.0, (w, h))
        for img in tagged_images:
            writer.write(img)


if __name__ == "__main__":
    main()
