import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import hydra
from dataclasses_json import DataClassJsonMixin, LetterCase, config
from omegaconf import DictConfig
from PIL import Image, ImageFile
from rhoknp import Document

from mdetr import BoundingBox, MDETRPrediction, predict_mdetr


class CamelCaseDataClassJsonMixin(DataClassJsonMixin):
    dataclasses_json_config = config(letter_case=LetterCase.CAMEL)['dataclasses_json']


@dataclass
class ImageInfo(CamelCaseDataClassJsonMixin):
    id: str
    path: str
    time: int


@dataclass
class UtteranceInfo(CamelCaseDataClassJsonMixin):
    text: str
    sids: list[str]
    start: int
    end: int
    duration: int
    speaker: str
    image_ids: list[str]


@dataclass
class DatasetInfo(CamelCaseDataClassJsonMixin):
    dataclass_json_config = dict(letter_case=LetterCase.CAMEL)

    scenario_id: str
    utterances: list[UtteranceInfo]
    images: list[ImageInfo]


@dataclass
class Phrase(CamelCaseDataClassJsonMixin):
    sid: str
    index: int
    text: str
    relation: str
    image: ImageInfo
    bounding_boxes: set[BoundingBox]


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
    image_dir = dataset_dir.joinpath('images')

    parsed_document: Document = run_cohesion(cfg.cohesion, input_knp_file)
    prediction_dir.joinpath(f'{parsed_document.did}.knp').write_text(parsed_document.to_knp())

    phrase_grounding_result = run_mdetr(cfg.mdetr, dataset_info, image_dir, parsed_document)
    prediction_dir.joinpath(f'{parsed_document.did}.json').write_text(phrase_grounding_result.to_json())


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


def run_mdetr(cfg: DictConfig, dataset_info: DatasetInfo, image_dir: Path, document: Document) -> PhraseGroundingResult:
    all_phrases: list[Phrase] = []
    sid2sentence = {sentence.sid: sentence for sentence in document.sentences}
    for utterance in dataset_info.utterances:
        corresponding_images = [image for image in dataset_info.images if image.id in utterance.image_ids]
        caption = Document.from_sentences([sid2sentence[sid] for sid in utterance.sids])
        for image in corresponding_images:
            phrases: list[Phrase] = []
            for base_phrase in caption.base_phrases:
                phrases.append(
                    Phrase(
                        sid=base_phrase.sentence.sid,
                        index=base_phrase.index,
                        text=base_phrase.text,
                        relation='=',
                        image=image,
                        bounding_boxes=set(),
                    )
                )
            image_file: ImageFile = Image.open(image_dir.joinpath(image.path))
            prediction: MDETRPrediction = predict_mdetr(cfg.checkpoint, image_file, caption)
            for bounding_box in prediction.bounding_boxes:
                for global_index, (word, prob) in enumerate(zip(prediction.words, bounding_box.word_probs)):
                    morpheme = caption.morphemes[global_index]
                    assert morpheme.text == word
                    if prob > 0.1:
                        phrases[global_index].bounding_boxes.add(bounding_box)
            all_phrases.extend(phrases)

    return PhraseGroundingResult(scenario_id=dataset_info.scenario_id, phrases=all_phrases)


if __name__ == '__main__':
    main()
