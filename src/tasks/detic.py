import math
import pickle
from pathlib import Path
from typing import Annotated

import luigi
import numpy as np
from omegaconf import DictConfig
from rhoknp import Document, Sentence

from tasks.detic_detection import DeticObjectDetection
from utils.prediction import BoundingBox as BoundingBoxPrediction
from utils.prediction import (
    PhraseGroundingPrediction,
    PhrasePrediction,
    RelationPrediction,
    UtterancePrediction,
)
from utils.util import DatasetInfo, Rectangle


class DeticPhraseGrounding(luigi.Task):
    scenario_id: Annotated[str, luigi.Parameter()] = luigi.Parameter()
    cfg: Annotated[DictConfig, luigi.Parameter()] = luigi.Parameter()
    document: Annotated[Document, luigi.Parameter()] = luigi.Parameter()
    dataset_dir: Annotated[Path, luigi.PathParameter()] = luigi.PathParameter()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        Path(self.cfg.prediction_dir).mkdir(exist_ok=True)

    def requires(self) -> luigi.Task:
        return DeticObjectDetection(scenario_id=self.scenario_id, cfg=self.cfg.detic_detection)

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
        with self.input().open(mode="r") as f:
            detection_dump: list[np.ndarray] = pickle.load(f)

        utterance_predictions: list[UtterancePrediction] = []
        dataset_info = DatasetInfo.from_json(self.dataset_dir.joinpath("info.json").read_text())
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
                    raw_bbs: np.ndarray = detection_dump[image_idx * 30]  # (bb, 6)
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
