import itertools
import math
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import luigi
from omegaconf import DictConfig
from rhoknp import Document, Sentence

from tasks.util import FileBasedResourceManagerMixin
from utils.mdetr import MDETRPrediction
from utils.prediction import BoundingBox as BoundingBoxPrediction
from utils.prediction import (
    PhraseGroundingPrediction,
    PhrasePrediction,
    RelationPrediction,
    UtterancePrediction,
)
from utils.util import DatasetInfo


class MDETRPhraseGrounding(luigi.Task, FileBasedResourceManagerMixin[int]):
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
        Path(self.cfg.prediction_dir).mkdir(exist_ok=True)

    def output(self):
        return luigi.LocalTarget(f"{self.cfg.prediction_dir}/{self.scenario_id}.json")

    def run(self):
        if (gpu_id := self.acquire_resource()) is None:
            raise RuntimeError("No available GPU.")
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            prediction = run_mdetr(
                self.cfg,
                dataset_dir=self.dataset_dir,
                document=Document.from_knp(self.document_path.read_text()),
                env=env,
            )
            with self.output().open(mode="w") as f:
                f.write(prediction.to_json(ensure_ascii=False, indent=2))
        except subprocess.CalledProcessError as e:
            print(e.stderr, file=sys.stderr)
            raise e
        finally:
            self.release_resource(gpu_id)


def run_mdetr(cfg: DictConfig, dataset_dir: Path, document: Document, env: dict[str, str]) -> PhraseGroundingPrediction:
    dataset_info = DatasetInfo.from_json(dataset_dir.joinpath("info.json").read_text())
    utterance_results: list[UtterancePrediction] = []
    sid2sentence: dict[str, Sentence] = {sentence.sid: sentence for sentence in document.sentences}
    for idx, utterance in enumerate(dataset_info.utterances):
        start_index = math.ceil(utterance.start / 1000)
        if idx + 1 < len(dataset_info.utterances):
            next_utterance = dataset_info.utterances[idx + 1]
            end_index = math.ceil(next_utterance.start / 1000)
        else:
            end_index = len(dataset_info.images)
        corresponding_images = dataset_info.images[start_index:end_index]
        caption = Document.from_sentences([sid2sentence[sid] for sid in utterance.sids])
        phrases: list[PhrasePrediction] = [
            PhrasePrediction(
                sid=base_phrase.sentence.sid,
                index=base_phrase.global_index,
                text=base_phrase.text,
                relations=[],
            )
            for base_phrase in caption.base_phrases
        ]
        with tempfile.TemporaryDirectory() as out_dir:
            caption_file = Path(out_dir).joinpath("caption.jpp")
            caption_file.write_text(caption.to_jumanpp())
            subprocess.run(
                [
                    cfg.python,
                    f"{cfg.project_root}/run_mdetr.py",
                    f"--model={cfg.checkpoint}",
                    f"--caption-file={caption_file}",
                    f"--backbone-name={cfg.backbone_name}",
                    f"--text-encoder={cfg.text_encoder}",
                    f"--batch-size={cfg.batch_size}",
                    f"--export-dir={out_dir}",
                    "--image-files",
                ]
                + [str(dataset_dir / image.path) for image in corresponding_images],
                check=True,
                env=env,
            )
            predictions = [MDETRPrediction.from_json(file.read_text()) for file in sorted(Path(out_dir).glob("*.json"))]

        assert len(corresponding_images) == len(predictions), f"{len(corresponding_images)} != {len(predictions)}"
        for (image, prediction), (phrase, base_phrase) in itertools.product(
            zip(corresponding_images, predictions),
            zip(phrases, caption.base_phrases),
        ):
            assert prediction.image_id == image.id
            for bounding_box in prediction.bounding_boxes:
                prob = max(bounding_box.word_probs[m.global_index] for m in base_phrase.morphemes)
                if prob >= 0.1:
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
        for phrase in phrases:
            phrase.relations.sort(key=lambda rel: rel.bounding_box.confidence, reverse=True)
        utterance_results.append(UtterancePrediction(text=caption.text, sids=utterance.sids, phrases=phrases))

    return PhraseGroundingPrediction(
        scenario_id=dataset_info.scenario_id, images=dataset_info.images, utterances=utterance_results
    )
