import argparse
import itertools
import math
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from operator import add
from pathlib import Path
from typing import Any

import polars as pl
from rhoknp import Document
from rhoknp.cohesion import ExophoraReferent

from cohesion_scorer import ScoreResult, SubScorer
from prediction_writer import PhraseGroundingPrediction
from utils.constants import CASES
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

    def eval_textual_reference(self, pred_document: Document) -> ScoreResult:
        scorer = SubScorer(
            pred_document,
            self.gold_document,
            cases=list(CASES),
            bridging=True,
            coreference=True,
            exophora_referents=[ExophoraReferent(e) for e in '著者 読者 不特定:人 不特定:物'.split()],
            pas=True,
        )
        score_result = scorer.run()
        return score_result

    def eval_visual_reference(self, prediction: PhraseGroundingPrediction, topk: int = -1) -> dict:
        recall_tracker = RatioTracker()
        precision_tracker = RatioTracker()
        # utterance ごとに評価
        sid2sentence = {sentence.sid: sentence for sentence in self.gold_document.sentences}
        assert len(self.dataset_info.utterances) == len(self.utterance_annotations) == len(prediction.utterances)
        all_image_ids = [image.id for image in self.dataset_info.images]
        for idx, (utterance, utterance_annotation, utterance_prediction) in enumerate(
            zip(self.dataset_info.utterances, self.utterance_annotations, prediction.utterances)
        ):
            base_phrases = [bp for sid in utterance.sids for bp in sid2sentence[sid].base_phrases]
            assert ''.join(bp.text for bp in base_phrases) == utterance_annotation.text == utterance_prediction.text
            start_index = math.ceil(utterance.start / 1000)
            if idx + 1 < len(self.dataset_info.utterances):
                next_utterance = self.dataset_info.utterances[idx + 1]
                end_index = math.ceil(next_utterance.start / 1000)
            else:
                end_index = len(all_image_ids)
            for image_id, (base_phrase, phrase_annotation, phrase_prediction) in itertools.product(
                all_image_ids[start_index:end_index],
                zip(base_phrases, utterance_annotation.phrases, utterance_prediction.phrases),
            ):
                # 対応する gold と system の BB を取得して比較
                sid = base_phrase.sentence.sid
                image_annotation: ImageAnnotation = self.image_id_to_annotation[image_id]
                instance_id_to_bounding_box: dict[str, BoundingBox] = {
                    bb.instance_id: bb for bb in image_annotation.bounding_boxes
                }
                assert base_phrase.text == phrase_annotation.text == phrase_prediction.text
                pred_relations = [rel for rel in phrase_prediction.relations if rel.image_id == image_id]
                # recall
                for gold_relation in phrase_annotation.relations:
                    # 現在の画像に含まれないオブジェクトは評価から除外
                    if gold_relation.instance_id not in instance_id_to_bounding_box:
                        continue
                    relation_type = gold_relation.type
                    gold_bounding_box = instance_id_to_bounding_box[gold_relation.instance_id]
                    gold_box: Rectangle = gold_bounding_box.rect
                    key = (image_id, sid, base_phrase.index, relation_type, gold_bounding_box.instance_id)
                    pred_bounding_boxes = [rel.bounding_box for rel in pred_relations if rel.type == relation_type]
                    if len(pred_bounding_boxes) > 0:
                        bounding_boxes = sorted(pred_bounding_boxes, key=lambda bb: bb.confidence, reverse=True)
                        if topk == -1:
                            pred_boxes = [
                                bb.rect for bb in bounding_boxes if bb.confidence >= self.confidence_threshold
                            ]
                        else:
                            pred_boxes = [bb.rect for bb in bounding_boxes[:topk]]
                        if any(box_iou(gold_box, pred_box) >= self.iou_threshold for pred_box in pred_boxes):
                            recall_tracker.add_positive(key[3])
                        else:
                            recall_tracker.add_negative(key[3])
                    else:
                        recall_tracker.add_negative(key[3])

                # precision
                for idx, pred_relation in enumerate(pred_relations):
                    relation_type = pred_relation.type
                    gold_relations = [rel for rel in phrase_annotation.relations if rel.type == relation_type]
                    if pred_relation.bounding_box.confidence < self.confidence_threshold:
                        continue
                    pred_box: Rectangle = pred_relation.bounding_box.rect
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
                        precision_tracker.add_positive(key[3])
                    if len(tp_gold_bounding_boxes) == 0:
                        key = (image_id, sid, base_phrase.index, relation_type, f'fp_{idx}')
                        precision_tracker.add_negative(key[3])
        eval_result: dict[str, dict[str, int]] = {}
        for rel in ('ガ', 'ヲ', 'ニ', 'ガ２', 'ノ', '='):
            eval_result[rel] = {
                'recall_pos': recall_tracker.positive[rel],
                'recall_total': recall_tracker.total[rel],
                'precision_pos': precision_tracker.positive[rel],
                'precision_total': precision_tracker.total[rel],
            }
        return eval_result


class RatioTracker:
    def __init__(self):
        self.total: dict[Any, int] = defaultdict(int)
        self.positive: dict[Any, int] = defaultdict(int)

    def add_positive(self, category: Any):
        self.total[category] += 1
        self.positive[category] += 1

    def add_negative(self, category: Any):
        self.total[category] += 1


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

    textual_results: list[ScoreResult] = []
    results = []
    for scenario_id in args.scenario_ids:
        dataset_info = DatasetInfo.from_json(args.dataset_dir.joinpath(f'{scenario_id}/info.json').read_text())
        gold_document = Document.from_knp(args.gold_knp_dir.joinpath(f'{scenario_id}.knp').read_text())
        pred_document = Document.from_knp(args.prediction_dir.joinpath(f'{scenario_id}.knp').read_text())
        image_text_annotation = ImageTextAnnotation.from_json(
            args.gold_image_dir.joinpath(f'{scenario_id}.json').read_text()
        )
        prediction = PhraseGroundingPrediction.from_json(
            args.prediction_dir.joinpath(f'{scenario_id}.json').read_text()
        )

        evaluator = MMRefEvaluator(
            dataset_info,
            gold_document,
            image_text_annotation,
        )
        textual_results.append(evaluator.eval_textual_reference(pred_document))
        for rel, measure in evaluator.eval_visual_reference(prediction, topk=args.recall_topk).items():
            result = {'scenario_id': scenario_id, 'rel': rel}
            result.update(measure)
            results.append(result)
    df = pl.DataFrame(results)
    df.drop_in_place('scenario_id')
    df_sum = df.groupby('rel', maintain_order=True).sum()
    # df = pl.concat([df, df_sum])
    df = df.with_columns(
        [
            (df['recall_pos'] / df['recall_total']).alias('recall'),
            (df['precision_pos'] / df['precision_total']).alias('precision'),
        ],
    )
    df_sum = df_sum.with_columns(
        [
            (df_sum['recall_pos'] / df_sum['recall_total']).alias('recall'),
            (df_sum['precision_pos'] / df_sum['precision_total']).alias('precision'),
        ]
    )
    print(df)
    print(df_sum)
    total_textual_result = reduce(add, textual_results)
    print(total_textual_result.to_dict())
    total_textual_result.export_csv('cohesion_result.csv')
    total_textual_result.export_txt('cohesion_result.txt')


if __name__ == '__main__':
    main()
