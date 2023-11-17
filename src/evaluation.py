import argparse
import itertools
import math
from collections import defaultdict
from functools import reduce
from operator import add
from pathlib import Path
from typing import Any

import polars as pl
from cohesion_tools.evaluation import CohesionScore, SubCohesionScorer
from rhoknp import Document
from rhoknp.cohesion import ExophoraReferent

from utils.annotation import BoundingBox, ImageAnnotation, ImageTextAnnotation
from utils.constants import CASES, RELATION_TYPES_ALL
from utils.prediction import BoundingBox as BoundingBoxPrediction
from utils.prediction import PhraseGroundingPrediction
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

    def eval_textual_reference(self, pred_document: Document) -> CohesionScore:
        scorer = SubCohesionScorer(
            pred_document,
            self.gold_document,
            exophora_referent_types=[ExophoraReferent(e).type for e in "著者 読者 不特定:人 不特定:物".split()],
            pas_cases=list(CASES),
            pas_verbal=True,
            pas_nominal=True,
            bridging=True,
            coreference=True,
        )
        return scorer.run()

    def eval_visual_reference(self, prediction: PhraseGroundingPrediction, topk: int = -1) -> list:
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
            assert "".join(bp.text for bp in base_phrases) == utterance_annotation.text == utterance_prediction.text
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
                    # 領域矩形は評価から除外
                    if gold_bounding_box.class_name == "region":
                        continue
                    key = (
                        image_id,
                        sid,
                        base_phrase.index,
                        relation_type,
                        gold_bounding_box.instance_id,
                        gold_bounding_box.class_name,
                    )
                    pred_bounding_boxes: list[BoundingBoxPrediction] = sorted(
                        {rel.bounding_box for rel in pred_relations if rel.type == relation_type},
                        key=lambda bb: bb.confidence,
                        reverse=True,
                    )
                    if len(pred_bounding_boxes) > 0:
                        gold_box: Rectangle = gold_bounding_box.rect
                        if topk == -1:
                            pred_boxes = [
                                bb.rect for bb in pred_bounding_boxes if bb.confidence >= self.confidence_threshold
                            ]
                        else:
                            pred_boxes = [bb.rect for bb in pred_bounding_boxes[:topk]]
                        if any(box_iou(gold_box, pred_box) >= self.iou_threshold for pred_box in pred_boxes):
                            recall_tracker.add_positive(key)
                        else:
                            recall_tracker.add_negative(key)
                    else:
                        recall_tracker.add_negative(key)

                # precision
                for rel_idx, pred_relation in enumerate(pred_relations):
                    relation_type = pred_relation.type
                    # allow ≒ relations
                    gold_relations = [
                        rel for rel in phrase_annotation.relations if rel.type in (relation_type, relation_type + "≒")
                    ]
                    if pred_relation.bounding_box.confidence < self.confidence_threshold:
                        continue
                    pred_box: Rectangle = pred_relation.bounding_box.rect
                    gold_bounding_boxes = [
                        instance_id_to_bounding_box[rel.instance_id]
                        for rel in gold_relations
                        if (
                            rel.instance_id in instance_id_to_bounding_box
                            and instance_id_to_bounding_box[rel.instance_id].class_name != "region"
                        )
                    ]
                    tp_gold_bounding_boxes = [
                        bb for bb in gold_bounding_boxes if box_iou(bb.rect, pred_box) >= self.iou_threshold
                    ]
                    for tp_gold_bounding_box in tp_gold_bounding_boxes:
                        key = (
                            image_id,
                            sid,
                            base_phrase.index,
                            relation_type,
                            tp_gold_bounding_box.instance_id,
                            tp_gold_bounding_box.class_name,
                        )
                        precision_tracker.add_positive(key)
                    if len(tp_gold_bounding_boxes) == 0:
                        key = (image_id, sid, base_phrase.index, relation_type, f"fp_{rel_idx}", "false_positive")
                        precision_tracker.add_negative(key)

        result_dict: dict[tuple[str, str, int, str, str, str], dict[str, Any]] = {}
        result_dict.update({key: {"recall_pos": value} for key, value in recall_tracker.positive.items()})
        for key, value in recall_tracker.total.items():
            if key in result_dict:
                result_dict[key]["recall_total"] = value
            else:
                result_dict[key] = {"recall_total": value}
        for key, value in precision_tracker.positive.items():
            if key in result_dict:
                result_dict[key]["precision_pos"] = value
            else:
                result_dict[key] = {"precision_pos": value}
        for key, value in precision_tracker.total.items():
            if key in result_dict:
                result_dict[key]["precision_total"] = value
            else:
                result_dict[key] = {"precision_total": value}
        results: list[dict[str, Any]] = []
        for key, metrics in result_dict.items():
            results.append(
                {
                    "image_id": key[0],
                    "sid": key[1],
                    "base_phrase_index": key[2],
                    "relation_type": key[3],
                    "instance_id": key[4],
                    "class_name": key[5],
                    "recall_pos": metrics.get("recall_pos", 0),
                    "recall_total": metrics.get("recall_total", 0),
                    "precision_pos": metrics.get("precision_pos", 0),
                    "precision_total": metrics.get("precision_total", 0),
                }
            )
        return results


class RatioTracker:
    def __init__(self):
        self.total: dict[Any, int] = defaultdict(int)
        self.positive: dict[Any, int] = defaultdict(int)

    def add_positive(self, category: Any):
        self.total[category] += 1
        self.positive[category] += 1

    def add_negative(self, category: Any):
        self.total[category] += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        "-d",
        type=Path,
        default="data/dataset",
        help="Path to the directory containing the target dataset.",
    )
    parser.add_argument("--gold-knp-dir", "-k", type=Path, default="data/knp", help="Path to the gold KNP directory.")
    parser.add_argument(
        "--gold-annotation-dir",
        "-a",
        type=Path,
        default="data/image_text_annotation",
        help="Path to the gold image text annotation file.",
    )
    parser.add_argument("--prediction-dir", "-p", type=Path, help="Path to the prediction directory.")
    parser.add_argument("--prediction-knp-dir", type=Path, help="Path to the prediction directory.")
    parser.add_argument("--scenario-ids", "--ids", type=str, nargs="*", help="List of scenario ids.")
    parser.add_argument("--recall-topk", "--topk", type=int, default=-1, help="For calculating Recall@k.")
    parser.add_argument(
        "--eval-modes", type=str, default="rel", nargs="*", choices=["rel", "class", "text"], help="evaluation modes"
    )
    parser.add_argument(
        "--format", type=str, default="repr", choices=["repr", "csv", "tsv", "json"], help="table format to print"
    )
    parser.add_argument("--raw-result-csv", type=Path, default=None, help="Path to the raw result csv file.")
    return parser.parse_args()


def main():
    args = parse_args()
    textual_results: list[CohesionScore] = []
    results = []
    for scenario_id in args.scenario_ids:
        dataset_info = DatasetInfo.from_json(args.dataset_dir.joinpath(f"{scenario_id}/info.json").read_text())
        gold_document = Document.from_knp(args.gold_knp_dir.joinpath(f"{scenario_id}.knp").read_text())
        pred_document = Document.from_knp(args.prediction_knp_dir.joinpath(f"{scenario_id}.knp").read_text())
        image_text_annotation = ImageTextAnnotation.from_json(
            args.gold_annotation_dir.joinpath(f"{scenario_id}.json").read_text()
        )
        prediction = PhraseGroundingPrediction.from_json(
            args.prediction_dir.joinpath(f"{scenario_id}.json").read_text()
        )

        evaluator = MMRefEvaluator(
            dataset_info,
            gold_document,
            image_text_annotation,
        )
        textual_results.append(evaluator.eval_textual_reference(pred_document))
        for row in evaluator.eval_visual_reference(prediction, topk=args.recall_topk):
            result = {"scenario_id": scenario_id}
            result.update(row)
            results.append(result)
    result_df = pl.DataFrame(results)
    if args.raw_result_csv is not None:
        result_df.write_csv(args.raw_result_csv)
    result_df.drop_in_place("scenario_id")

    if "rel" in args.eval_modes:
        df_rel = (
            result_df.groupby("relation_type", maintain_order=True)
            .sum()
            .drop(["image_id", "sid", "base_phrase_index", "instance_id", "class_name"])
        )
        df_rel = df_rel.with_columns(
            [
                (df_rel["recall_pos"] / df_rel["recall_total"]).alias("recall"),
                (df_rel["precision_pos"] / df_rel["precision_total"]).alias("precision"),
            ]
        )
        # sort dataframe by relation type
        df_rel = (
            df_rel.with_columns(
                df_rel["relation_type"].apply(lambda x: RELATION_TYPES_ALL.index(x)).alias("case_index")
            )
            .sort("case_index")
            .drop("case_index")
        )
        print(df_to_string(df_rel, args.format))

    if "class" in args.eval_modes:
        df_class = (
            result_df.filter(pl.col("relation_type") == "=")
            .groupby("class_name", maintain_order=True)
            .sum()
            .drop(["image_id", "sid", "base_phrase_index", "relation_type", "instance_id"])
        )
        df_class = df_class.with_columns(
            [
                (df_class["recall_pos"] / df_class["recall_total"]).alias("recall"),
                (df_class["precision_pos"] / df_class["precision_total"]).alias("precision"),
            ]
        )
        print(df_to_string(df_class, args.format))

    if "text" in args.eval_modes:
        total_textual_result = reduce(add, textual_results)
        total_textual_result.export_csv("cohesion_result.csv")
        total_textual_result.export_txt("cohesion_result.txt")


def df_to_string(df: pl.DataFrame, format_: str) -> str:
    if format_ == "repr":
        pl.Config.set_tbl_rows(100)
        pl.Config.set_tbl_cols(16)
        return str(df)
    elif format_ == "csv":
        return df.write_csv()
    elif format_ == "tsv":
        return df.write_csv(separator="\t")
    elif format_ == "json":
        return df.write_json()
    else:
        raise ValueError(f"Unknown format: {format_}")


if __name__ == "__main__":
    main()
