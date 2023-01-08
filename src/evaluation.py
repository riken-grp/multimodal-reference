import argparse
import itertools
from pathlib import Path

from rhoknp import Document

from prediction_writer import PhraseGroundingResult, PhraseResult
from utils.image import BoundingBox, ImageAnnotation, ImageTextAnnotation
from utils.util import DatasetInfo, Rectangle, box_iou


class MMRefEvaluator:
    def __init__(self, dataset_info: DatasetInfo, gold_document: Document, image_text_annotation: ImageTextAnnotation):
        assert dataset_info.scenario_id == gold_document.doc_id == image_text_annotation.scenario_id
        self.dataset_info = dataset_info
        self.gold_document = gold_document
        self.utterance_annotations = image_text_annotation.utterances
        self.image_id_to_annotation: dict[str, ImageAnnotation] = {
            image.image_id: image for image in image_text_annotation.images
        }
        self.confidence_threshold = 0.8
        self.iou_threshold = 0.5

    def eval_textual_reference(self, result: PhraseGroundingResult) -> dict:
        raise NotImplementedError

    def eval_visual_reference(self, result: PhraseGroundingResult) -> dict:
        result_dict = {}
        # utterance ごとに評価
        sid2sentence = {sentence.sid: sentence for sentence in self.gold_document.sentences}
        assert len(self.dataset_info.utterances) == len(self.utterance_annotations) == len(result.utterances)
        for utterance, utterance_annotation, utterance_result in zip(
            self.dataset_info.utterances, self.utterance_annotations, result.utterances
        ):
            document = Document.from_sentences([sid2sentence[sid] for sid in utterance.sids])
            assert document.text == utterance_annotation.text
            for image_id, phrase in itertools.product(utterance.image_ids, document.base_phrases):
                # 対応する gold と system の BB を取得して比較
                sid = phrase.sentence.sid
                image_annotation: ImageAnnotation = self.image_id_to_annotation[image_id]
                instance_id_to_bounding_box: dict[str, BoundingBox] = {
                    bb.instance_id: bb for bb in image_annotation.bounding_boxes
                }
                phrase_annotation = utterance_annotation.phrases[phrase.global_index]
                assert phrase_annotation.text == phrase.text
                pred_phrases: list[PhraseResult] = list(
                    filter(
                        lambda p: p.image.id == image_id and p.sid == sid and p.index == phrase.index,
                        utterance_result.phrases,
                    )
                )
                # recall
                for gold_relation in phrase_annotation.relations:
                    relation_type = gold_relation.type
                    gold_bounding_box = instance_id_to_bounding_box[gold_relation.instance_id]
                    gold_box: Rectangle = gold_bounding_box.rect
                    for pred_phrase in pred_phrases:
                        if pred_phrase.relation != relation_type:
                            continue
                        pred_boxes = [
                            bb.rect for bb in pred_phrase.bounding_boxes if bb.confidence >= self.confidence_threshold
                        ]
                        key = (image_id, sid, phrase.index, relation_type, gold_bounding_box.instance_id)
                        if any(box_iou(gold_box, pred_box) >= self.iou_threshold for pred_box in pred_boxes):
                            result_dict[key] = 'tp'
                        else:
                            result_dict[key] = 'fn'
                # precision
                for pred_phrase in pred_phrases:
                    relation_type = pred_phrase.relation
                    gold_relations = [rel for rel in phrase_annotation.relations if rel.type == relation_type]
                    for pred_bounding_box in pred_phrase.bounding_boxes:
                        if pred_bounding_box.confidence < self.confidence_threshold:
                            continue
                        pred_box: Rectangle = pred_bounding_box.rect
                        gold_bounding_boxes = [instance_id_to_bounding_box[rel.instance_id] for rel in gold_relations]
                        for gold_bounding_box in gold_bounding_boxes:
                            key = (image_id, sid, phrase.index, relation_type, gold_bounding_box.instance_id)
                            gold_box = gold_bounding_box.rect
                            if box_iou(gold_box, pred_box) >= self.iou_threshold:
                                assert result_dict[key] == 'tp'
                            else:
                                result_dict[key] = 'fp'
        eval_result = {}
        for rel in ('ガ', 'ヲ', 'ニ', 'ノ', '='):
            vs = [v for k, v in result_dict.items() if k[3] == rel]
            tp = vs.count('tp')
            fn = vs.count('fn')
            fp = vs.count('fp')
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            eval_result[rel] = {'precision': precision, 'recall': recall, 'f1': f1}
        return eval_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', '-d', type=Path, help='Path to the directory containing the target dataset.')
    parser.add_argument('--gold-text-file', '-t', type=Path, help='Path to the gold KNP file.')
    parser.add_argument('--gold-image-file', '-i', type=Path, help='Path to the gold image text annotation file.')
    parser.add_argument('--prediction-file', '-p', type=Path, help='Path to the prediction file.')
    args = parser.parse_args()

    dataset_info = DatasetInfo.from_json(args.dataset_dir.joinpath('info.json').read_text())
    gold_document = Document.from_knp(args.gold_text_file.read_text())
    image_text_annotation = ImageTextAnnotation.from_json(args.gold_image_file.read_text())
    prediction = PhraseGroundingResult.from_json(args.prediction_file.read_text())

    evaluator = MMRefEvaluator(
        dataset_info,
        gold_document,
        image_text_annotation,
    )
    # print(evaluator.eval_textual_reference(utterance, document))
    print(evaluator.eval_visual_reference(prediction))


if __name__ == '__main__':
    main()
