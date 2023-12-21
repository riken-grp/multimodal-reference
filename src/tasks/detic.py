import math
import os
import pickle
import socket
import subprocess
from pathlib import Path
from typing import Annotated

import luigi
import numpy as np
from omegaconf import DictConfig
from rhoknp import Document, Sentence

from tasks.util import FileBasedResourceManagerMixin
from utils.prediction import BoundingBox as BoundingBoxPrediction
from utils.prediction import (
    PhraseGroundingPrediction,
    PhrasePrediction,
    RelationPrediction,
    UtterancePrediction,
)
from utils.util import DatasetInfo, Rectangle


class DeticPhraseGrounding(luigi.Task, FileBasedResourceManagerMixin[int]):
    scenario_id: Annotated[str, luigi.Parameter()] = luigi.Parameter()
    cfg: Annotated[DictConfig, luigi.Parameter()] = luigi.Parameter()
    document: Annotated[Document, luigi.Parameter()] = luigi.Parameter()
    dataset_dir: Annotated[Path, luigi.PathParameter()] = luigi.PathParameter()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        available_gpus = [int(gpu_id) for gpu_id in os.environ.get("AVAILABLE_GPUS", "0").split(",")]
        super(luigi.Task, self).__init__(
            available_gpus, Path("shared_state.json"), state_prefix=f"{socket.gethostname()}_gpu"
        )
        Path(self.cfg.prediction_dir).mkdir(exist_ok=True)

    def output(self):
        return luigi.LocalTarget(f"{self.cfg.prediction_dir}/{self.scenario_id}.json")

    def run(self):
        dataset_info = DatasetInfo.from_json(self.dataset_dir.joinpath("info.json").read_text())
        cfg = self.cfg
        dump_file = Path(cfg.dump_dir) / f"{self.scenario_id}.npy"
        dump_file.parent.mkdir(exist_ok=True, parents=True)
        input_video_file = Path(cfg.recording_dir) / self.scenario_id / "fp_video.mp4"

        gpu_id = self.acquire_resource()
        if gpu_id is None:
            raise RuntimeError("No available GPU.")
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            subprocess.run(
                [
                    cfg.python,
                    f"{cfg.project_root}/export.py",
                    f"--config-file={cfg.config}",
                    f"--video-input={input_video_file.resolve()}",
                    "--vocabulary=lvis",
                    f"--output={dump_file.resolve()}",
                    "--opts",
                    "MODEL.WEIGHTS",
                    cfg.model,
                ],
                cwd=cfg.project_root,
                env=env,
                check=True,
            )
        finally:
            self.release_resource(gpu_id)

        with dump_file.open(mode="rb") as f:
            detic_dump: list[np.ndarray] = pickle.load(f)

        utterance_predictions: list[UtterancePrediction] = []
        sid2sentence: dict[str, Sentence] = {sentence.sid: sentence for sentence in self.document.sentences}
        for idx, utterance in enumerate(dataset_info.utterances):
            start_index = math.ceil(utterance.start / 1000)
            if idx + 1 < len(dataset_info.utterances):
                next_utterance = dataset_info.utterances[idx + 1]
                end_index = math.ceil(next_utterance.start / 1000)
            else:
                end_index = len(dataset_info.images)
            corresponding_images = dataset_info.images[start_index:end_index]
            caption = Document.from_sentences([sid2sentence[sid] for sid in utterance.sids])
            phrase_predictions: list[PhrasePrediction] = [
                PhrasePrediction(
                    sid=base_phrase.sentence.sid,
                    index=base_phrase.global_index,
                    text=base_phrase.text,
                    relations=[],
                )
                for base_phrase in caption.base_phrases
            ]
            for phrase_prediction in phrase_predictions:
                for image in corresponding_images:
                    image_idx = int(image.id) - 1
                    raw_bbs: np.ndarray = detic_dump[image_idx * 30]  # (bb, 6)
                    for raw_bb in raw_bbs.tolist():
                        phrase_prediction.relations.append(
                            RelationPrediction(
                                type="=",
                                image_id=image.id,
                                bounding_box=BoundingBoxPrediction(
                                    image_id=image.id,
                                    rect=Rectangle(
                                        x1=raw_bb[0] / 2, y1=raw_bb[1] / 2, x2=raw_bb[2] / 2, y2=raw_bb[3] / 2
                                    ),
                                    confidence=raw_bb[4],
                                ),
                            )
                        )

            utterance_predictions.append(
                UtterancePrediction(text=caption.text, sids=utterance.sids, phrases=phrase_predictions)
            )

        prediction = PhraseGroundingPrediction(
            scenario_id=self.scenario_id,
            images=dataset_info.images,
            utterances=utterance_predictions,
        )

        with self.output().open(mode="w") as f:
            f.write(prediction.to_json(ensure_ascii=False, indent=2))
