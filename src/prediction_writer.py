import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import hydra
from omegaconf import DictConfig
from PIL import Image, ImageFile
from rhoknp import Document

from mdetr import BoundingBox, MDETRPrediction, predict_mdetr
from utils import CamelCaseDataClassJsonMixin, DatasetInfo, ImageInfo


@dataclass
class Phrase(CamelCaseDataClassJsonMixin):
    sid: str
    index: int
    text: str
    relation: str
    image: ImageInfo
    bounding_boxes: list[BoundingBox]


@dataclass
class PhraseGroundingResult(CamelCaseDataClassJsonMixin):
    scenario_id: str
    phrases: list[Phrase]


@hydra.main(version_base=None, config_path="../configs")
def main(cfg: DictConfig) -> None:

    dataset_dir = Path(cfg.dataset_dir)
    prediction_dir = Path(cfg.prediction_dir)

    dataset_info = DatasetInfo.from_json(dataset_dir.joinpath('info.json').read_text())
    input_knp_file = dataset_dir.joinpath('raw.knp')

    parsed_document: Document = run_cohesion(cfg.cohesion, input_knp_file)
    prediction_dir.joinpath(f'{parsed_document.did}.knp').write_text(parsed_document.to_knp())

    phrase_grounding_result = run_mdetr(cfg.mdetr, dataset_info, dataset_dir, parsed_document)
    prediction_dir.joinpath(f'{parsed_document.did}.json').write_text(
        phrase_grounding_result.to_json(ensure_ascii=False, indent=2)
    )


def run_cohesion(cfg: DictConfig, input_knp_file: Path) -> Document:
    with tempfile.TemporaryDirectory() as out_dir:
        subprocess.run(
            [
                cfg.python,
                f'{cfg.project_root}/src/predict.py',
                f'checkpoint={cfg.checkpoint}',
                f'input_path={input_knp_file}',
                f'export_dir={out_dir}',
            ]
        )
        return Document.from_knp(next(Path(out_dir).glob('*.knp')).read_text())


def run_mdetr(
    cfg: DictConfig, dataset_info: DatasetInfo, dataset_dir: Path, document: Document
) -> PhraseGroundingResult:
    all_phrases: list[Phrase] = []
    sid2sentence = {sentence.sid: sentence for sentence in document.sentences}
    for utterance in dataset_info.utterances:
        corresponding_images = [image for image in dataset_info.images if image.id in utterance.image_ids]
        caption = Document.from_sentences([sid2sentence[sid] for sid in utterance.sids])
        for image in corresponding_images:
            phrases: list[Phrase] = [
                Phrase(
                    sid=base_phrase.sentence.sid,
                    index=base_phrase.index,
                    text=base_phrase.text,
                    relation='=',
                    image=image,
                    bounding_boxes=[],
                )
                for base_phrase in caption.base_phrases
            ]
            image_file: ImageFile = Image.open(dataset_dir.joinpath(image.path))
            prediction: MDETRPrediction = predict_mdetr(cfg.checkpoint, image_file, caption)
            for bounding_box in prediction.bounding_boxes:
                for phrase, base_phrase in zip(phrases, caption.base_phrases):
                    words = [prediction.words[m.global_index] for m in base_phrase.morphemes]
                    assert ''.join(words) == phrase.text == base_phrase.text
                    prob = max(bounding_box.word_probs[m.global_index] for m in base_phrase.morphemes)
                    if prob >= 0.1:
                        phrase.bounding_boxes.append(bounding_box)
            all_phrases.extend(filter(lambda p: p.bounding_boxes, phrases))

    return PhraseGroundingResult(scenario_id=dataset_info.scenario_id, phrases=all_phrases)


if __name__ == '__main__':
    main()
