import argparse
import itertools
from pathlib import Path

from rhoknp import Document

from prediction_writer import PhraseGroundingResult
from utils.image import ImageAnnotation
from utils.util import DatasetInfo


class MMRefEvaluator:
    def __init__(self, dataset_info: DatasetInfo, gold_document: Document, image_annotations: list[ImageAnnotation]):
        self.dataset_info = dataset_info
        self.gold_document = gold_document
        self.gold_image_id_to_annotation: dict[str, ImageAnnotation] = {ann.image_id: ann for ann in image_annotations}

    def eval_textual_reference(self, result: PhraseGroundingResult) -> dict:
        raise NotImplementedError

    def eval_visual_reference(self, result: PhraseGroundingResult) -> dict:
        # utterance ごとに評価
        sid2sentence = {sentence.sid: sentence for sentence in self.gold_document.sentences}
        assert len(self.dataset_info.utterances) == len(result.utterances)
        for utterance, utterance_result in zip(self.dataset_info.utterances, result.utterances):
            document = Document.from_sentences([sid2sentence[sid] for sid in utterance.sids])
            for image_id, phrase in itertools.product(utterance.image_ids, document.base_phrases):
                # 対応する gold と system の BB を取得して比較
                gold_image_annotation: ImageAnnotation = self.gold_image_id_to_annotation[image_id]
                pred_phrases = list(
                    filter(
                        lambda p: p.image.id == image_id and p.sid == phrase.sentence.sid and p.index == phrase.index,
                        utterance_result.phrases,
                    )
                )
                print(gold_image_annotation, pred_phrases)
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', '-d', type=Path, help='Path to the directory containing the target dataset.')
    parser.add_argument('--gold-text-file', '-t', type=Path, help='Path to the gold KNP file.')
    parser.add_argument('--gold-image-dir', '-i', type=Path, help='Path to the directory containing image annotations.')
    parser.add_argument('--prediction-file', '-p', type=Path, help='Path to the prediction file.')
    args = parser.parse_args()

    dataset_info: DatasetInfo = DatasetInfo.from_json(args.dataset_dir.joinpath('info.json').read_text())
    gold_document: Document = Document.from_knp(args.gold_text_file.read_text())
    image_annotations: list[ImageAnnotation] = []
    for image_annotation_path in sorted(args.gold_image_dir.glob('.png.json')):
        image_annotations.append(ImageAnnotation.from_json(image_annotation_path.read_text()))
    prediction: PhraseGroundingResult = PhraseGroundingResult.from_json(args.prediction_file.read_text())

    evaluator = MMRefEvaluator(
        dataset_info,
        gold_document,
        image_annotations,
    )
    # print(evaluator.eval_textual_reference(utterance, document))
    print(evaluator.eval_visual_reference(prediction))


if __name__ == '__main__':
    main()
