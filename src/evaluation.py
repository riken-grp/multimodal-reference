import argparse
import itertools
from dataclasses import dataclass
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
        self.confidence_threshold = 0.9
        self.iou_threshold = 0.5

    def eval_textual_reference(self, result: PhraseGroundingResult) -> dict:
        raise NotImplementedError

    def eval_visual_reference(self, result: PhraseGroundingResult, topk: int = -1) -> dict:
        result_dict = {}
        # utterance ごとに評価
        sid2sentence = {sentence.sid: sentence for sentence in self.gold_document.sentences}
        assert len(self.dataset_info.utterances) == len(self.utterance_annotations) == len(result.utterances)
        for utterance, utterance_annotation, utterance_result in zip(
            self.dataset_info.utterances, self.utterance_annotations, result.utterances
        ):
            base_phrases = [bp for sid in utterance.sids for bp in sid2sentence[sid].base_phrases]
            assert ''.join(bp.text for bp in base_phrases) == utterance_annotation.text
            for image_id, (base_phrase, phrase_annotation) in itertools.product(
                utterance.image_ids, zip(base_phrases, utterance_annotation.phrases)
            ):
                # 対応する gold と system の BB を取得して比較
                sid = base_phrase.sentence.sid
                image_annotation: ImageAnnotation = self.image_id_to_annotation[image_id]
                instance_id_to_bounding_box: dict[str, BoundingBox] = {
                    bb.instance_id: bb for bb in image_annotation.bounding_boxes
                }
                assert phrase_annotation.text == base_phrase.text
                pred_phrases: list[PhraseResult] = list(
                    filter(
                        lambda p: p.image.id == image_id and p.sid == sid and p.index == base_phrase.index,
                        utterance_result.phrases,
                    )
                )
                # recall
                for gold_relation in phrase_annotation.relations:
                    # 現在の画像に含まれないオブジェクトは評価から除外
                    if gold_relation.instance_id not in instance_id_to_bounding_box:
                        continue
                    relation_type = gold_relation.type
                    gold_bounding_box = instance_id_to_bounding_box[gold_relation.instance_id]
                    gold_box: Rectangle = gold_bounding_box.rect
                    key = (image_id, sid, base_phrase.index, relation_type, gold_bounding_box.instance_id)
                    rel_pred_phrases = [p for p in pred_phrases if p.relation == relation_type]
                    assert len(rel_pred_phrases) in (0, 1)
                    if len(rel_pred_phrases) == 1:
                        pred_phrase = rel_pred_phrases[0]
                        bounding_boxes = sorted(pred_phrase.bounding_boxes, key=lambda bb: bb.confidence, reverse=True)
                        if topk == -1:
                            pred_boxes = [
                                bb.rect for bb in bounding_boxes if bb.confidence >= self.confidence_threshold
                            ]
                        else:
                            pred_boxes = [bb.rect for bb in bounding_boxes[:topk]]
                        if any(box_iou(gold_box, pred_box) >= self.iou_threshold for pred_box in pred_boxes):
                            result_dict[key] = 'tp'
                        else:
                            result_dict[key] = 'fn'
                    else:
                        result_dict[key] = 'fn'

                # precision
                for pred_phrase in pred_phrases:
                    relation_type = pred_phrase.relation
                    gold_relations = [rel for rel in phrase_annotation.relations if rel.type == relation_type]
                    for idx, pred_bounding_box in enumerate(pred_phrase.bounding_boxes):
                        if pred_bounding_box.confidence < self.confidence_threshold:
                            continue
                        pred_box: Rectangle = pred_bounding_box.rect
                        gold_bounding_boxes = [
                            instance_id_to_bounding_box[rel.instance_id]
                            for rel in gold_relations
                            if rel.instance_id in instance_id_to_bounding_box
                        ]
                        tp_gold_bounding_boxes = [
                            bb for bb in gold_bounding_boxes if box_iou(bb.rect, pred_box) >= self.iou_threshold
                        ]
                        for tp_gold_bounding_box in tp_gold_bounding_boxes:
                            key = (image_id, sid, base_phrase.index, relation_type, tp_gold_bounding_box.instance_id)
                            assert result_dict[key] == 'tp'
                        if len(tp_gold_bounding_boxes) == 0:
                            key = (image_id, sid, base_phrase.index, relation_type, f'fp_{idx}')
                            result_dict[key] = 'fp'
        eval_result = {}
        for rel in ('ガ', 'ヲ', 'ニ', 'ノ', '='):
            vs = [v for k, v in result_dict.items() if k[3] == rel]
            tp = vs.count('tp')
            fn = vs.count('fn')
            fp = vs.count('fp')
            measure = Measure(correct=tp, denom_gold=tp + fn, denom_pred=tp + fp)
            eval_result[rel] = measure
        return eval_result


@dataclass
class Measure:
    """A data class to calculate and represent F-measure"""

    correct: int = 0
    denom_gold: int = 0
    denom_pred: int = 0

    def __add__(self, other: 'Measure'):
        return Measure(
            self.denom_pred + other.denom_pred, self.denom_gold + other.denom_gold, self.correct + other.correct
        )

    @property
    def precision(self) -> float:
        if self.denom_pred == 0:
            return 0.0
        return self.correct / self.denom_pred

    @property
    def recall(self) -> float:
        if self.denom_gold == 0:
            return 0.0
        return self.correct / self.denom_gold

    @property
    def f1(self) -> float:
        if self.denom_pred + self.denom_gold == 0:
            return 0.0
        return 2 * self.correct / (self.denom_pred + self.denom_gold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', '-d', type=Path, help='Path to the directory containing the target dataset.')
    parser.add_argument('--gold-knp-dir', '-k', type=Path, help='Path to the gold KNP directory.')
    parser.add_argument('--gold-image-dir', '-i', type=Path, help='Path to the gold image text annotation file.')
    parser.add_argument('--prediction-dir', '-p', type=Path, help='Path to the prediction directory.')
    parser.add_argument('--scenario-ids', type=str, nargs='*', help='List of scenario ids.')
    parser.add_argument('--recall-topk', '--topk', type=int, default=-1, help='For calculating Recall@k.')
    args = parser.parse_args()

    for scenario_id in args.scenario_ids:

        dataset_info = DatasetInfo.from_json(args.dataset_dir.joinpath(f'{scenario_id}/info.json').read_text())
        gold_document = Document.from_knp(args.gold_knp_dir.joinpath(f'{scenario_id}.knp').read_text())
        image_text_annotation = ImageTextAnnotation.from_json(
            args.gold_image_dir.joinpath(f'{scenario_id}.json').read_text()
        )
        prediction = PhraseGroundingResult.from_json(args.prediction_dir.joinpath(f'{scenario_id}.json').read_text())

        evaluator = MMRefEvaluator(
            dataset_info,
            gold_document,
            image_text_annotation,
        )
        # print(evaluator.eval_textual_reference(utterance, document))
        print(evaluator.eval_visual_reference(prediction, topk=args.recall_topk))


if __name__ == '__main__':
    main()
