import math
import os
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import luigi
from omegaconf import DictConfig
from rhoknp import KNP, Document, Sentence

from tasks.util import FileBasedResourceManagerMixin
from utils.glip import GLIPPrediction
from utils.prediction import BoundingBox as BoundingBoxPrediction
from utils.prediction import (
    PhraseGroundingPrediction,
    PhrasePrediction,
    RelationPrediction,
    UtterancePrediction,
)
from utils.util import DatasetInfo


class GLIPPhraseGrounding(luigi.Task, FileBasedResourceManagerMixin[int]):
    scenario_id: Annotated[str, luigi.Parameter()] = luigi.Parameter()
    cfg: Annotated[DictConfig, luigi.Parameter()] = luigi.Parameter()
    document_path: Annotated[Path, luigi.Parameter()] = luigi.PathParameter()
    dataset_dir: Annotated[Path, luigi.PathParameter()] = luigi.PathParameter()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        available_gpus = [int(gpu_id) for gpu_id in os.environ.get("AVAILABLE_GPUS", "0").split(",")]
        super(luigi.Task, self).__init__(
            available_gpus, Path("shared_state.json"), state_prefix=f"{socket.gethostname()}_gpu"
        )
        Path(self.cfg.prediction_dir).mkdir(parents=True, exist_ok=True)

    def output(self):
        return luigi.LocalTarget(f"{self.cfg.prediction_dir}/{self.scenario_id}.json")

    def run(self):
        if (gpu_id := self.acquire_resource()) is None:
            raise RuntimeError("No available GPU.")
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            prediction = run_glip(
                self.cfg,
                dataset_dir=self.dataset_dir,
                document=Document.from_knp(self.document_path.read_text()),
                env=env,
            )
            with self.output().open(mode="w") as f:
                f.write(prediction.to_json(ensure_ascii=False, indent=2))
        finally:
            self.release_resource(gpu_id)


def run_glip(cfg: DictConfig, dataset_dir: Path, document: Document, env: dict[str, str]) -> PhraseGroundingPrediction:
    dataset_info = DatasetInfo.from_json(dataset_dir.joinpath("info.json").read_text())
    utterance_predictions: list[UtterancePrediction] = []
    sid2sentence: dict[str, Sentence] = {sentence.sid: sentence for sentence in document.sentences}
    for idx, utterance in enumerate(dataset_info.utterances):
        start_index = math.ceil(utterance.start / 1000)
        if idx + 1 < len(dataset_info.utterances):
            next_utterance = dataset_info.utterances[idx + 1]
            end_index = math.ceil(next_utterance.start / 1000)
        else:
            end_index = len(dataset_info.images)
        images_in_utterance = dataset_info.images[start_index:end_index]
        utterances_in_window = dataset_info.utterances[max(0, idx + 1 - cfg.num_utterances_in_window) : idx + 1]
        doc_window = Document.from_sentences(
            [sid2sentence[sid] for utterance in utterances_in_window for sid in utterance.sids]
        )
        doc_utterance = Document.from_sentences([sid2sentence[sid] for sid in utterance.sids])
        phrases: list[PhrasePrediction] = [
            PhrasePrediction(
                sid=base_phrase.sentence.sid,
                index=base_phrase.global_index,
                text=base_phrase.text,
                relations=[],
            )
            for base_phrase in doc_utterance.base_phrases
        ]
        with tempfile.TemporaryDirectory() as out_dir:
            caption_file = Path(out_dir).joinpath("caption.knp")
            if cfg.no_query is False:
                caption_file.write_text(doc_window.to_knp())
            else:
                caption_file.write_text(KNP().apply(Sentence.from_raw_text("もの")).to_knp())
            subprocess.run(
                [
                    cfg.python,
                    f"{cfg.project_root}/tools/run_glip.py",
                    f"--model={cfg.checkpoint}",
                    f"--config-file={cfg.config}",
                    f"--caption-file={caption_file}",
                    f"--export-dir={out_dir}",
                    "--image-files",
                ]
                + [str(dataset_dir / image.path) for image in images_in_utterance],
                check=True,
                env=env,
            )
            predictions = [GLIPPrediction.from_json(file.read_text()) for file in sorted(Path(out_dir).glob("*.json"))]

        assert len(images_in_utterance) == len(predictions), f"{len(images_in_utterance)} != {len(predictions)}"
        for image, prediction in zip(images_in_utterance, predictions):
            for phrase in phrases:
                if cfg.no_query is False:
                    phrase_prediction = prediction.phrases[phrase.index + len(prediction.phrases) - len(phrases)]
                    assert phrase_prediction.text == phrase.text
                else:
                    assert len(prediction.phrases) == 1
                    phrase_prediction = prediction.phrases[0]
                for bounding_box in phrase_prediction.bounding_boxes:
                    assert bounding_box.image_id == image.id
                    phrase.relations.append(
                        RelationPrediction(
                            type="=",
                            image_id=image.id,
                            bounding_box=BoundingBoxPrediction(
                                image_id=image.id,
                                rect=bounding_box.rect,
                                confidence=bounding_box.confidence,
                            ),
                        )
                    )
        utterance_predictions.append(UtterancePrediction(text=doc_utterance.text, sids=utterance.sids, phrases=phrases))

    return PhraseGroundingPrediction(
        scenario_id=dataset_info.scenario_id, images=dataset_info.images, utterances=utterance_predictions
    )
