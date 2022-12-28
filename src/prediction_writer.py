import argparse
from dataclasses import dataclass
from pathlib import Path

from dataclasses_json import LetterCase, dataclass_json
from rhoknp import KWJA, Document

from mdetr import BoundingBox, MDETRPrediction, predict_mdetr


@dataclass_json
@dataclass
class ImageInfo:
    id: str
    path: str
    time: int


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class UtteranceInfo:
    text: str
    sids: list[str]
    start: int
    end: int
    duration: int
    speaker: str
    image_ids: list[str]


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class DatasetInfo:
    scenario_id: str
    utterances: list[UtteranceInfo]
    images: list[ImageInfo]


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class Phrase:
    sid: str
    index: int
    text: str
    relation: str
    image: ImageInfo
    bounding_boxes: set[BoundingBox]


@dataclass_json(letter_case=LetterCase.CAMEL)
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
            image_path = image_dir.joinpath(image.path)
            prediction: MDETRPrediction = predict_mdetr(mdetr_checkpoint_path, image_path, caption)
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
