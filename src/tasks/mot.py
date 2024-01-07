import json
import os
import pickle
import socket
from pathlib import Path
from typing import Annotated, Union

import cv2
import hydra
import luigi
import numpy as np
import torch
from boxmot import OCSORT, BoTSORT, BYTETracker, DeepOCSORT, HybridSORT, StrongSORT
from omegaconf import DictConfig
from typing_extensions import TypeAlias

from tasks.util import FileBasedResourceManagerMixin
from utils.mot import BoundingBox, DetectionLabels, Frame
from utils.util import Rectangle

Tracker: TypeAlias = Union[BoTSORT, StrongSORT, HybridSORT, DeepOCSORT, OCSORT, BYTETracker]


class MultipleObjectTracking(luigi.Task, FileBasedResourceManagerMixin[int]):
    scenario_id: Annotated[str, luigi.Parameter()] = luigi.Parameter()
    cfg: Annotated[DictConfig, luigi.Parameter()] = luigi.Parameter()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        available_gpus = [int(gpu_id) for gpu_id in os.environ.get("AVAILABLE_GPUS", "0").split(",")]
        super(luigi.Task, self).__init__(
            available_gpus, Path("shared_state.json"), state_prefix=f"{socket.gethostname()}_gpu"
        )
        Path(self.cfg.prediction_dir).mkdir(exist_ok=True, parents=True)

    def requires(self) -> luigi.Task:
        return hydra.utils.instantiate(self.cfg.detection, scenario_id=self.scenario_id)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(f"{self.cfg.prediction_dir}/{self.scenario_id}.json")

    def complete(self) -> bool:
        if not Path(self.output().path).exists():
            return False

        self_mtime = Path(self.output().path).stat().st_mtime
        task = self.requires()
        if not task.complete():
            return False
        output = task.output()
        assert isinstance(output, luigi.LocalTarget), f"output is not LocalTarget: {output}"
        if Path(output.path).stat().st_mtime > self_mtime:
            return False

        return True

    def run(self) -> None:
        cfg = self.cfg
        input_video_file = Path(cfg.recording_dir) / self.scenario_id / "fp_video.mp4"

        if (gpu_id := self.acquire_resource()) is None:
            raise RuntimeError("No available GPU.")
        try:
            if hasattr(cfg.tracker, "device"):
                cfg.tracker.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
            tracker: Tracker = hydra.utils.instantiate(cfg.tracker)
            with self.input().open(mode="r") as f:
                detection_dump: list[np.ndarray] = pickle.load(f)
            class_names: list[str] = json.loads(Path("lvis_categories.json").read_text())
            detection_labels = run_tracker(tracker, input_video_file, detection_dump, class_names)
            with self.output().open(mode="w") as f:
                f.write(detection_labels.to_json(ensure_ascii=False, indent=2))
        finally:
            self.release_resource(gpu_id)


def frame_from_video(video: cv2.VideoCapture):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


def run_tracker(
    tracker: Tracker, input_video_file: Path, detection_dump: list[np.ndarray], class_names: list[str]
) -> DetectionLabels:
    video = cv2.VideoCapture(str(input_video_file))

    frames = []
    frame: np.ndarray  # (h, w, 3), BGR
    for idx, frame in enumerate(frame_from_video(video)):
        if idx >= len(detection_dump):
            break
        raw_bbs: np.ndarray = detection_dump[idx]  # (bb, 6), the 2nd axis: (x1, y1, x2, y2, confidence, class_id)
        if len(raw_bbs.shape) != 2 or raw_bbs.shape[1] != 6:
            raw_bbs = np.empty((0, 6))
        # filter out too small BBs
        raw_bbs = raw_bbs[raw_bbs[:, 2] - raw_bbs[:, 0] >= 1]
        raw_bbs = raw_bbs[raw_bbs[:, 3] - raw_bbs[:, 1] >= 1]

        tracked_bbs: np.ndarray = tracker.update(raw_bbs, frame)  # (bb, 7)

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

        frames.append(Frame(index=idx, bounding_boxes=bounding_boxes))

    return DetectionLabels(frames=frames, class_names=class_names)
