import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from boxmot import StrongSORT
from cv2 import cv2

from utils.util import CamelCaseDataClassJsonMixin


@dataclass(frozen=True)
class BoundingBox(CamelCaseDataClassJsonMixin):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    instance_id: int

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1


@dataclass(frozen=True)
class Frame(CamelCaseDataClassJsonMixin):
    bounding_boxes: list[BoundingBox]


@dataclass(frozen=True)
class DetectionLabels(CamelCaseDataClassJsonMixin):
    frames: list[Frame]
    classes: list[str]


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
        model_weights=Path("resnet50_msmt17.pt"),
        device="cuda" if torch.cuda.is_available() else "cpu",
        fp16=False,
    )

    # Detection Model
    with open(args.detic_dump, mode="rb") as f:
        detic_dump: list[np.ndarray] = pickle.load(f)
    class_names: list[str] = json.loads(Path("lvis_categories.json").read_text())

    colors: np.ndarray = (np.random.rand(len(class_names) * 3) * 255).astype(np.uint8).reshape(-1, 3)  # (names, 3)

    video = cv2.VideoCapture(str(args.video))

    tagged_images = []
    frames = []
    for idx, frame in enumerate(frame_from_video(video)):
        frame: np.ndarray  # (h, w, 3)
        if idx >= len(detic_dump):
            break
        # print(detic_dump[idx][1])  # xyxy, confidence, class for image 0 in the batch
        raw_bbs: np.ndarray = detic_dump[idx]  # (bb, 6)
        if len(raw_bbs.shape) != 2 or raw_bbs.shape[1] != 6:
            raw_bbs = np.empty((0, 6))

        tracked_bbs: np.ndarray = mot_tracker.update(torch.as_tensor(raw_bbs), frame)  # (bb, 7)

        bounding_boxes: list[BoundingBox] = []
        for tracked_bb in tracked_bbs:  # xyxy, confidence, class_id, instance_id for image 0 in the batch
            x1, y1, x2, y2, instance_id, class_id, confidence = tracked_bb.tolist()
            x1, y1, x2, y2, class_id, instance_id = int(x1), int(y1), int(x2), int(y2), int(class_id), int(instance_id)
            color: list[int] = colors[class_id].tolist()
            cv2.rectangle(frame, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=5)
            cv2.putText(
                frame,
                f"{class_names[class_id]}_{instance_id}",
                (x1, y1),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=5,
                thickness=5,
                color=(255, 255, 255),
            )
            bounding_boxes.append(BoundingBox(x1, y1, x2, y2, confidence, class_id, instance_id))

        if args.show:
            cv2.imshow("img", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        tagged_images.append(frame)
        frames.append(Frame(bounding_boxes))

    if args.output_video is not None:
        fourcc: int = cv2.VideoWriter_fourcc(*"xvid")
        w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(args.output_video), fourcc, 30.0, (w, h))
        for img in tagged_images:
            writer.write(img)

    if args.output_json is not None:
        args.output_json.write_text(DetectionLabels(frames, class_names).to_json(indent=2))


if __name__ == "__main__":
    main()
