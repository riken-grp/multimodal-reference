import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from boxmot import StrongSORT

from utils.mot import BoundingBox, DetectionLabels, Frame
from utils.util import Rectangle


def parse_args():
    parser = argparse.ArgumentParser(description="Tracking Sample")
    parser.add_argument("video", help="Input video file", type=Path)
    parser.add_argument("--output-video", default=None, help="Output video file", type=Path)
    parser.add_argument("--show", action="store_true", help="Show video while processing")
    parser.add_argument("--output-json", default=None, help="Output json file", type=Path)
    parser.add_argument("--detic-dump", default=None, help="Detic detection result pickle dump file", type=str)
    return parser.parse_args()


def frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


def main():
    args = parse_args()

    # Tracker
    mot_tracker = StrongSORT(
        model_weights=Path("model/osnet_ain_x1_0_msmt17.pt"),
        device="cuda" if torch.cuda.is_available() else "cpu",
        fp16=False,
    )

    # Detection Model
    with open(args.detic_dump, mode="rb") as f:
        detic_dump: list[np.ndarray] = pickle.load(f)
    class_names: list[str] = json.loads(Path("lvis_categories.json").read_text())

    rng = np.random.default_rng(1337)
    colors: np.ndarray = (rng.random(len(class_names) * 3) * 255).astype(np.uint8).reshape(-1, 3)  # (names, 3)

    video = cv2.VideoCapture(str(args.video))

    tagged_images = []
    frames = []
    frame: np.ndarray  # (h, w, 3), BGR
    for idx, frame in enumerate(frame_from_video(video)):
        if idx >= len(detic_dump):
            break
        raw_bbs: np.ndarray = detic_dump[idx]  # (bb, 6), the 2nd axis: (x1, y1, x2, y2, confidence, class_id)
        if len(raw_bbs.shape) != 2 or raw_bbs.shape[1] != 6:
            raw_bbs = np.empty((0, 6))
        # filter out too small BBs
        raw_bbs = raw_bbs[raw_bbs[:, 2] - raw_bbs[:, 0] >= 1]
        raw_bbs = raw_bbs[raw_bbs[:, 3] - raw_bbs[:, 1] >= 1]

        tracked_bbs: np.ndarray = mot_tracker.update(raw_bbs, frame)  # (bb, 7)

        bounding_boxes: list[BoundingBox] = []
        for tracked_bb in tracked_bbs:
            # https://github.com/mikel-brostrom/yolo_tracking#custom-object-detection-model-example
            # confidence は detection の confidence そのまま
            x1, y1, x2, y2, instance_id, confidence, class_id, _ = tracked_bb.tolist()
            x1, y1, x2, y2, class_id, instance_id = int(x1), int(y1), int(x2), int(y2), int(class_id), int(instance_id)
            bounding_boxes.append(
                BoundingBox(
                    rect=Rectangle(x1, y1, x2, y2),
                    confidence=confidence,
                    class_name=class_names[class_id],
                    instance_id=instance_id,
                )
            )

        if args.show:
            for bounding_box in bounding_boxes:
                rect = bounding_box.rect
                color: list[int] = colors[class_names.index(bounding_box.class_name)].tolist()
                cv2.rectangle(frame, pt1=(rect.x1, rect.y1), pt2=(rect.x2, rect.y2), color=color, thickness=5)
                cv2.putText(
                    frame,
                    f"{bounding_box.class_name}_{bounding_box.instance_id}",
                    (rect.x1, rect.y1),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=5,
                    thickness=5,
                    color=(255, 255, 255),
                )
            cv2.imshow("img", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        tagged_images.append(frame)
        frames.append(Frame(index=idx, bounding_boxes=bounding_boxes))

    if args.output_video is not None:
        fourcc: int = cv2.VideoWriter_fourcc(*"xvid")
        w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(args.output_video), fourcc, 30.0, (w, h))
        for img in tagged_images:
            writer.write(img)

    if args.output_json is not None:
        args.output_json.write_text(DetectionLabels(frames=frames, class_names=class_names).to_json(indent=2))


if __name__ == "__main__":
    main()
