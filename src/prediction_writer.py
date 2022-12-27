import argparse
from dataclasses import dataclass
from pathlib import Path

from dataclasses_json import dataclass_json
from rhoknp import KWJA, Document

from mdetr import BoundingBox, MDETRPrediction, predict_mdetr


@dataclass_json
@dataclass
class ImageInfo:
    id: str
    path: str
    time: int


@dataclass_json
@dataclass
class UtteranceInfo:
    text: str
    sids: list[str]
    start: int
    end: int
    duration: int
    speaker: str
    image_ids: list[str]


@dataclass_json
@dataclass
class DatasetInfo:
    scenario_id: str
    utterances: list[UtteranceInfo]
    images: list[ImageInfo]


@dataclass_json
@dataclass
class Phrase:
    sid: str
    phrase_id: int
    text: str
    relation: str
    image: ImageInfo
    bounding_boxes: list[BoundingBox]


@dataclass_json
@dataclass
class PhraseGroundingResult:
    scenario_id: str
    phrases: list[Phrase]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdetr-checkpoint', type=Path)
    parser.add_argument('--dataset-dir', type=Path)
    parser.add_argument('--prediction-dir', type=Path)
    args = parser.parse_args()

    dataset_info = DatasetInfo.from_json(args.dataset_dir.joinpath('info.json').read_text())
    document = Document.from_knp(args.dataset_dir.joinpath('raw.knp').read_text())
    image_dir = args.dataset_dir.joinpath('images')

    kwja = KWJA()
    parsed_document: Document = kwja.apply_to_document(document)
    args.prediction_dir.joinpath(f'{parsed_document.did}.knp').write_text(parsed_document.to_knp())

    phrase_grounding_result = run_mdetr(args.mdetr_checkpoint, dataset_info, image_dir, parsed_document)
    args.prediction_dir.joinpath(f'{parsed_document.did}.json').write_text(phrase_grounding_result.to_json())


def run_mdetr(
    mdetr_checkpoint_path: Path, dataset_info: DatasetInfo, image_dir: Path, document: Document
) -> PhraseGroundingResult:
    phrases: list[Phrase] = []
    sid2sentence = {sentence.sid: sentence for sentence in document.sentences}
    for utterance in dataset_info.utterances:
        corresponding_images = [image for image in dataset_info.images if image.id in utterance.image_ids]
        caption = Document.from_sentences([sid2sentence[sid] for sid in utterance.sids])
        for image in corresponding_images:
            image_path = image_dir.joinpath(image.path)
            prediction: MDETRPrediction = predict_mdetr(mdetr_checkpoint_path, image_path, caption)
            print(prediction)

    return PhraseGroundingResult(scenario_id=dataset_info.scenario_id, phrases=phrases)


if __name__ == '__main__':
    main()
