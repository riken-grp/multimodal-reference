import argparse
import itertools
import math
import pickle
from collections import defaultdict
from functools import reduce
from operator import add
from pathlib import Path
from typing import Any, Literal, Optional

import motmetrics
import numpy as np
import polars as pl
from cohesion_tools.evaluators import CohesionEvaluator, CohesionScore
from rhoknp import Document
from rhoknp.cohesion import ExophoraReferent

from utils.annotation import BoundingBox, ImageAnnotation, ImageTextAnnotation
from utils.constants import CASES, RELATION_TYPES_ALL
from utils.mot import DetectionLabels
from utils.prediction import BoundingBox as BoundingBoxPrediction
from utils.prediction import PhraseGroundingPrediction
from utils.util import DatasetInfo, IdMapper, Rectangle, UtteranceInfo, box_iou, image_id_to_msec


class MMRefEvaluator:
    def __init__(self, dataset_info: DatasetInfo, gold_document: Document, image_text_annotation: ImageTextAnnotation):
        assert dataset_info.scenario_id == gold_document.doc_id == image_text_annotation.scenario_id
        self.dataset_info = dataset_info
        self.gold_document = gold_document
        self.utterance_annotations = image_text_annotation.utterances
        self.image_annotations: list[ImageAnnotation] = image_text_annotation.images
        self.iou_threshold = 0.5
        self.cohesion_evaluator = CohesionEvaluator(
            tasks=["pas", "bridging", "coreference"],
            exophora_referent_types=[ExophoraReferent(e).type for e in "著者 読者 不特定:人 不特定:物".split()],
            pas_cases=list(CASES),
            bridging_rel_types=["ノ"],
        )
        self.cohesion_evaluator.coreference_evaluator.is_target_mention = (
            lambda mention: mention.features.get("体言") is True
        )

    def eval_textual_reference(self, pred_document: Document) -> CohesionScore:
        return self.cohesion_evaluator.run(
            predicted_documents=[pred_document],
            gold_documents=[self.gold_document],
        )

    def eval_object_detection(
        self,
        pred_detection: np.ndarray,
        recall_top_ks: list[int],
        confidence_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        recall_predictions: dict[tuple, list] = {}
        precision_predictions: dict[tuple, list] = {}
        for image_annotation in self.image_annotations:
            image_idx = int(image_annotation.image_id) - 1
            raw_bbs: np.ndarray = pred_detection[image_idx * 30]
            predicted_bounding_boxes = [
                BoundingBoxPrediction(
                    image_id=image_annotation.image_id,
                    rect=Rectangle(x1=raw_bb[0] / 2, y1=raw_bb[1] / 2, x2=raw_bb[2] / 2, y2=raw_bb[3] / 2),
                    confidence=raw_bb[4],
                )
                for raw_bb in raw_bbs.tolist()
            ]
            # Recall
            for gold_bounding_box in image_annotation.bounding_boxes:
                if gold_bounding_box.class_name == "region":
                    continue
                # `key` identifies each gold bounding box
                key: tuple = (
                    self.dataset_info.scenario_id,
                    image_annotation.image_id,
                    gold_bounding_box.instance_id,
                    -1,
                    gold_bounding_box.class_name,
                    -1,
                )
                recall_predictions[key] = [
                    (pred_bounding_box.confidence, box_iou(gold_bounding_box.rect, pred_bounding_box.rect))
                    for pred_bounding_box in predicted_bounding_boxes
                ]

            # Precision
            for pred_idx, pred_bounding_box in enumerate(predicted_bounding_boxes):
                # `key` identifies each predicted bounding box
                key = (
                    self.dataset_info.scenario_id,
                    image_annotation.image_id,
                    "",
                    pred_idx,
                    "",
                    pred_bounding_box.confidence,
                )
                precision_predictions[key] = [
                    box_iou(gold_bounding_box.rect, pred_bounding_box.rect)
                    for gold_bounding_box in image_annotation.bounding_boxes
                    if gold_bounding_box.class_name != "region"
                ]

        results: list[dict[str, Any]] = []
        for key, orig_preds in recall_predictions.items():
            comps: list[tuple[float, bool]] = sorted(orig_preds, key=lambda x: x[0], reverse=True)
            if confidence_threshold >= 0:
                comps = [comp for comp in comps if comp[0] >= confidence_threshold]
            recall_pos_dict = {}
            for recall_top_k in recall_top_ks:
                top_k_comps = comps[:recall_top_k] if recall_top_k >= 0 else comps
                recall_pos = int(any(comp[1] >= self.iou_threshold for comp in top_k_comps))  # 0 if comps is empty
                recall_pos_dict[f"recall_pos@{recall_top_k}" if recall_top_k >= 0 else "recall_pos"] = recall_pos
            results.append(
                {
                    "scenario_id": key[0],
                    "image_id": key[1],
                    "instance_id": key[2],
                    "pred_idx": key[3],
                    "class_name": key[4],
                    "precision_pos": 0,
                    "precision_total": 0,
                    "recall_total": 1,
                }
                | recall_pos_dict
            )

        for key, ious in precision_predictions.items():
            confidence: float = key[5]
            if confidence < confidence_threshold:
                continue
            recall_pos_dict = {f"recall_pos@{recall_top_k}": 0 for recall_top_k in recall_top_ks}
            results.append(
                {
                    "scenario_id": key[0],
                    "image_id": key[1],
                    "instance_id": key[2],
                    "pred_idx": key[3],
                    "class_name": key[4],
                    "precision_pos": int(any(iou >= self.iou_threshold for iou in ious)),
                    "precision_total": 1,
                    "recall_total": 0,
                }
                | recall_pos_dict
            )

        return results

    def eval_mot(self, pred_mot: DetectionLabels) -> motmetrics.MOTAccumulator:
        accumulator = motmetrics.MOTAccumulator(auto_id=True)  # `auto_id=True`: Automatically increment frame id
        id_mapper = IdMapper()

        pred_frames = [pred_mot.frames[i] for i in range(0, len(pred_mot.frames), 30)]
        assert len(pred_frames) == len(self.image_annotations)
        for pred_frame, image_annotation in zip(pred_frames, self.image_annotations):
            distance_matrix = motmetrics.distances.iou_matrix(
                [bb.rect.to_xywh() for bb in image_annotation.bounding_boxes],  # Ground truth objects in this frame
                [
                    (bb.rect.x1 / 2, bb.rect.y1 / 2, bb.rect.w / 2, bb.rect.h / 2) for bb in pred_frame.bounding_boxes
                ],  # Detector hypotheses in this frame
                max_iou=0.5,
            )
            accumulator.update(
                [id_mapper[bb.instance_id] for bb in image_annotation.bounding_boxes],
                [bb.instance_id for bb in pred_frame.bounding_boxes],
                distance_matrix,
            )

        return accumulator

    def eval_visual_reference(
        self,
        prediction: PhraseGroundingPrediction,
        recall_top_ks: list[int],
        confidence_threshold: float = 0.0,
        image_span: Literal["past-current", "prev-current", "current", "prev-next", "current-next"] = "current-next",
    ) -> list[dict[str, Any]]:
        recall_rects, precision_rects = self._compare_prediction_and_annotation(prediction, image_span=image_span)

        result_dict: dict[tuple[str, str, int, str, str, str], dict[str, Any]] = defaultdict(dict)

        for key, orig_rects in recall_rects.items():
            rects: list[tuple[float, bool]] = sorted(orig_rects, key=lambda x: x[0], reverse=True)
            if confidence_threshold >= 0:
                rects = [rect for rect in rects if rect[0] >= confidence_threshold]
            for recall_top_k in recall_top_ks:
                top_k_rects = rects[:recall_top_k] if recall_top_k >= 0 else rects
                recall_pos = int(any(rect[1] >= self.iou_threshold for rect in top_k_rects))  # 0 if rects is empty
                result_dict[key][f"recall_pos@{recall_top_k}"] = recall_pos
            result_dict[key]["recall_total"] = 1

        for key, rects in precision_rects.items():
            confidence: float = next(iter(rects))[0]
            if confidence < confidence_threshold:
                continue
            # old way of calculating precision
            # precision_pos += sum(rect[1] for rect in rects)
            # precision_total += max(1, sum(rect[1] for rect in rects))
            result_dict[key]["precision_pos"] = int(
                any(rect[1] >= self.iou_threshold for rect in rects)
            )  # 0 if rects is empty
            result_dict[key]["precision_total"] = 1

        results: list[dict[str, Any]] = []
        for key, metrics in result_dict.items():
            results.append(
                {
                    "scenario_id": self.dataset_info.scenario_id,
                    "image_id": key[0],
                    "utterance_id": key[1],
                    "sid": key[2],
                    "base_phrase_index": key[3],
                    "rel_type": key[4],
                    "instance_id_or_pred_idx": key[5],
                    "class_name": key[6],
                    "width": key[7],
                    "height": key[8],
                    "center_x": key[9],
                    "center_y": key[10],
                    "pos": key[11],
                    "subpos": key[12],
                    "temporal_location": key[13],
                    "recall_total": metrics.get("recall_total", 0),
                    "precision_pos": metrics.get("precision_pos", 0),
                    "precision_total": metrics.get("precision_total", 0),
                }
                | {
                    "recall_pos"
                    + (f"@{recall_top_k}" if recall_top_k >= 0 else ""): metrics.get(f"recall_pos@{recall_top_k}", 0)
                    for recall_top_k in recall_top_ks
                }
            )
        return results

    def _compare_prediction_and_annotation(
        self,
        prediction: PhraseGroundingPrediction,
        image_span: Literal["past-current", "prev-current", "current", "prev-next", "current-next"],
    ) -> tuple[dict, dict]:
        recall_rects: dict[tuple[Any, ...], list[tuple[float, float]]] = {}
        precision_rects: dict[tuple[Any, ...], list[tuple[float, float]]] = {}

        # utterance ごとに評価
        sid2sentence = {sentence.sid: sentence for sentence in self.gold_document.sentences}
        image_id_to_annotation: dict[str, ImageAnnotation] = {image.image_id: image for image in self.image_annotations}
        assert len(self.dataset_info.utterances) == len(self.utterance_annotations) == len(prediction.utterances)
        all_image_ids = [image.id for image in self.dataset_info.images]
        utterance: UtteranceInfo
        for idx, (utterance, utterance_annotation, utterance_prediction) in enumerate(
            zip(self.dataset_info.utterances, self.utterance_annotations, prediction.utterances)
        ):
            base_phrases = [bp for sid in utterance.sids for bp in sid2sentence[sid].base_phrases]
            assert "".join(bp.text for bp in base_phrases) == utterance_annotation.text == utterance_prediction.text
            start_index = math.ceil(utterance.start / 1000)
            end_index = math.ceil(utterance.end / 1000)
            if idx >= 1:
                prev_utterance = self.dataset_info.utterances[idx - 1]
                prev_end_index = math.ceil(prev_utterance.end / 1000)
            else:
                prev_end_index = 0
            if idx + 1 < len(self.dataset_info.utterances):
                next_utterance = self.dataset_info.utterances[idx + 1]
                next_start_index = math.ceil(next_utterance.start / 1000)
            else:
                next_start_index = len(all_image_ids)
            if image_span == "past-current":
                image_ids = all_image_ids[:end_index]
            elif image_span == "prev-current":
                image_ids = all_image_ids[prev_end_index:end_index]
            elif image_span == "current":
                image_ids = all_image_ids[start_index:end_index]
            elif image_span == "prev-next":
                image_ids = all_image_ids[prev_end_index:next_start_index]
            elif image_span == "current-next":
                image_ids = all_image_ids[start_index:next_start_index]
            else:
                raise ValueError(f"Unknown image_span: {image_span}")
            for image_id, (base_phrase, phrase_annotation, phrase_prediction) in itertools.product(
                image_ids,
                zip(base_phrases, utterance_annotation.phrases, utterance_prediction.phrases),
            ):
                # 対応する gold と system の BB を取得して比較
                sid = base_phrase.sentence.sid
                image_annotation: ImageAnnotation = image_id_to_annotation[image_id]
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
                    frame_msec = image_id_to_msec(image_id)
                    if frame_msec < utterance.start:
                        temporal_location = "before"
                    elif utterance.start <= frame_msec < (utterance.start + utterance.end) / 2:
                        temporal_location = "first_half"
                    elif (utterance.start + utterance.end) / 2 <= frame_msec < utterance.end:
                        temporal_location = "second_half"
                    elif utterance.end <= frame_msec:
                        temporal_location = "after"
                    else:
                        raise ValueError(f"Unknown temporal location: {frame_msec} for utterance {utterance}")
                    key = (
                        image_id,
                        f"{utterance.speaker}_{idx:02}",
                        sid,
                        base_phrase.index,
                        relation_type,
                        gold_bounding_box.instance_id,
                        gold_bounding_box.class_name,
                        gold_bounding_box.rect.w,
                        gold_bounding_box.rect.h,
                        gold_bounding_box.rect.cx,
                        gold_bounding_box.rect.cy,
                        base_phrase.head.pos,
                        base_phrase.head.subpos if base_phrase.head.text != "物" else "形式名詞",
                        temporal_location,
                    )
                    pred_bounding_boxes: list[BoundingBoxPrediction] = sorted(
                        {rel.bounding_box for rel in pred_relations if rel.type == relation_type},
                        key=lambda bb: bb.confidence,
                        reverse=True,
                    )
                    recall_rects[key] = [
                        (pred_bounding_box.confidence, box_iou(gold_bounding_box.rect, pred_bounding_box.rect))
                        for pred_bounding_box in pred_bounding_boxes
                    ]

                # precision
                for rel_idx, pred_relation in enumerate(pred_relations):
                    relation_type = pred_relation.type
                    # allow ≒ relations
                    gold_relations = [
                        rel for rel in phrase_annotation.relations if rel.type in (relation_type, relation_type + "≒")
                    ]
                    pred_rect: Rectangle = pred_relation.bounding_box.rect
                    gold_bounding_boxes = [
                        instance_id_to_bounding_box[rel.instance_id]
                        for rel in gold_relations
                        if (
                            rel.instance_id in instance_id_to_bounding_box
                            and instance_id_to_bounding_box[rel.instance_id].class_name != "region"
                        )
                    ]
                    frame_msec = image_id_to_msec(image_id)
                    if frame_msec < utterance.start:
                        temporal_location = "before"
                    elif utterance.start <= frame_msec < (utterance.start + utterance.end) / 2:
                        temporal_location = "first_half"
                    elif (utterance.start + utterance.end) / 2 <= frame_msec < utterance.end:
                        temporal_location = "second_half"
                    elif utterance.end <= frame_msec:
                        temporal_location = "after"
                    else:
                        raise ValueError(f"Unknown temporal location: {frame_msec} for utterance {utterance}")
                    key = (
                        image_id,
                        f"{utterance.speaker}_{idx:02}",
                        sid,
                        base_phrase.index,
                        relation_type,
                        str(rel_idx),
                        "",
                        pred_rect.w,
                        pred_rect.h,
                        pred_rect.cx,
                        pred_rect.cy,
                        base_phrase.head.pos,
                        base_phrase.head.subpos if base_phrase.head.text != "物" else "形式名詞",
                        temporal_location,
                    )
                    rects = [
                        (pred_relation.bounding_box.confidence, box_iou(gold_bounding_box.rect, pred_rect))
                        for gold_bounding_box in gold_bounding_boxes
                    ]
                    if not rects:
                        # Ensure at least one rect to store confidence
                        rects.append((pred_relation.bounding_box.confidence, -1.0))
                    precision_rects[key] = rects

        return recall_rects, precision_rects


def parse_args() -> argparse.Namespace:
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
    parser.add_argument(
        "--prediction-mmref-dir", "-p", type=Path, default="result/mmref", help="Path to the prediction directory."
    )
    parser.add_argument(
        "--prediction-knp-dir", type=Path, default="result/cohesion", help="Path to the prediction directory."
    )
    parser.add_argument(
        "--prediction-mot-dir", type=Path, default="result/mot", help="Path to the prediction directory."
    )
    parser.add_argument(
        "--prediction-detection-dir",
        type=Path,
        default="result/detection/th0.5",
        help="Path to the prediction directory.",
    )
    parser.add_argument("--scenario-ids", "--ids", type=str, nargs="*", required=True, help="List of scenario ids.")
    parser.add_argument("--recall-topk", "--topk", type=int, nargs="*", default=[], help="For calculating Recall@k.")
    parser.add_argument(
        "--confidence-threshold",
        "--th",
        type=float,
        default=0.0,
        help="Confidence threshold for predicted bounding boxes.",
    )
    parser.add_argument(
        "--eval-modes",
        type=str,
        default=["rel"],
        nargs="+",
        choices=["rel", "class", "size", "noun", "temporal", "text", "mot", "detection"],
        help="Evaluation modes.",
    )
    parser.add_argument(
        "--format", type=str, default="repr", choices=["repr", "csv", "tsv", "json"], help="table format to print"
    )
    parser.add_argument("--raw-result-csv", type=Path, default=None, help="Path to the raw result csv file.")
    parser.add_argument("--column-prefixes", type=str, nargs="*", default=None, help="Path to the raw result csv file.")
    parser.add_argument(
        "--image-span",
        type=str,
        default="current-next",
        choices=["past-current", "prev-current", "current", "prev-next", "current-next"],
        help="Image span to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_results: dict[str, list[Any]] = defaultdict(list)
    for scenario_id in args.scenario_ids:
        dataset_info = DatasetInfo.from_json(args.dataset_dir.joinpath(f"{scenario_id}/info.json").read_text())
        gold_document = Document.from_knp(args.gold_knp_dir.joinpath(f"{scenario_id}.knp").read_text())
        image_text_annotation = ImageTextAnnotation.from_json(
            args.gold_annotation_dir.joinpath(f"{scenario_id}.json").read_text()
        )
        evaluator = MMRefEvaluator(dataset_info, gold_document, image_text_annotation)

        if "text" in args.eval_modes:
            pred_document = Document.from_knp(args.prediction_knp_dir.joinpath(f"{scenario_id}.knp").read_text())
            eval_results["text"].append(evaluator.eval_textual_reference(pred_document))
        if "mot" in args.eval_modes:
            pred_mot = DetectionLabels.from_json(args.prediction_mot_dir.joinpath(f"{scenario_id}.json").read_text())
            eval_results["mot"].append(evaluator.eval_mot(pred_mot))
        if "detection" in args.eval_modes:
            with args.prediction_detection_dir.joinpath(f"{scenario_id}.npy").open(mode="rb") as f:
                pred_detection: np.ndarray = pickle.load(f)
            eval_results["detection"] += evaluator.eval_object_detection(
                pred_detection, recall_top_ks=args.recall_topk, confidence_threshold=args.confidence_threshold
            )
        if set(args.eval_modes) & {"rel", "class", "size", "noun", "temporal"}:
            prediction = PhraseGroundingPrediction.from_json(
                args.prediction_mmref_dir.joinpath(f"{scenario_id}.json").read_text()
            )
            eval_results["mmref"] += evaluator.eval_visual_reference(
                prediction,
                recall_top_ks=args.recall_topk,
                confidence_threshold=args.confidence_threshold,
                image_span=args.image_span,
            )

    if "text" in args.eval_modes:
        textual_reference_result = reduce(add, eval_results["text"])
        textual_reference_result.export_csv("textual_reference_result.csv")
        textual_reference_result.export_txt("textual_reference_result.txt")

    if "mot" in args.eval_modes:
        metrics_host: motmetrics.metrics.MetricsHost = motmetrics.metrics.create()
        summary = metrics_host.compute_many(
            eval_results["mot"],
            metrics=["idf1", "idp", "idr", "recall", "precision", "mota", "motp"],
            names=args.scenario_ids,
            generate_overall=True,
        )
        print(df_to_string(pl.from_pandas(summary.tail(1)), args.format))

    if set(args.eval_modes) & {"rel", "class", "size", "noun", "temporal"}:
        mmref_result_df = pl.DataFrame(eval_results["mmref"])
        if args.raw_result_csv is not None:
            mmref_result_df.write_csv(args.raw_result_csv)

        if "rel" in args.eval_modes:
            print_relation_table(mmref_result_df, args.recall_topk, args.column_prefixes, args.format)

        if "class" in args.eval_modes:
            print_class_table(mmref_result_df, args.recall_topk, args.column_prefixes, args.format)

        if "size" in args.eval_modes:
            print_size_table(mmref_result_df, args.recall_topk, args.column_prefixes, args.format)

        if "noun" in args.eval_modes:
            print_noun_table(mmref_result_df, args.recall_topk, args.column_prefixes, args.format)

        if "temporal" in args.eval_modes:
            print_temporal_table(mmref_result_df, args.recall_topk, args.column_prefixes, args.format)

    if "detection" in args.eval_modes:
        detection_result_df = pl.DataFrame(eval_results["detection"])
        if args.raw_result_csv is not None:
            detection_result_df.write_csv(args.raw_result_csv)
        num_images = detection_result_df.group_by("scenario_id", "image_id").count().shape[0]
        detection_result_df = detection_result_df.sum().drop(
            "scenario_id", "image_id", "instance_id", "pred_idx", "class_name"
        )
        new_columns = [
            (detection_result_df["precision_pos"] / detection_result_df["precision_total"]).alias("precision")
        ]
        for recall_top_k in args.recall_topk:
            metric_suffix = f"@{recall_top_k}" if recall_top_k >= 0 else ""
            new_columns.append(
                (detection_result_df["recall_pos" + metric_suffix] / detection_result_df["recall_total"]).alias(
                    "recall" + metric_suffix
                )
            )
        detection_result_df = detection_result_df.with_columns(new_columns).with_columns(num_images=num_images)
        print(df_to_string(detection_result_df, args.format))


def print_relation_table(
    mmref_result_df: pl.DataFrame,
    recall_top_ks: list[int],
    column_prefixes: list[str],
    format_: str = "repr",
) -> None:
    df_rel = (
        mmref_result_df.group_by("rel_type", maintain_order=True)
        .sum()
        .drop(
            [
                "scenario_id",
                "image_id",
                "utterance_id",
                "sid",
                "base_phrase_index",
                "instance_id_or_pred_idx",
                "class_name",
            ]
        )
    )

    new_columns = [(df_rel["precision_pos"] / df_rel["precision_total"]).alias("precision")]
    for recall_top_k in recall_top_ks:
        metric_suffix = f"@{recall_top_k}" if recall_top_k >= 0 else ""
        new_columns.append(
            (df_rel["recall_pos" + metric_suffix] / df_rel["recall_total"]).alias("recall" + metric_suffix)
        )
    df_rel = df_rel.with_columns(new_columns)
    # sort dataframe by relation type
    df_rel = (
        df_rel.with_columns(df_rel["rel_type"].map_elements(lambda x: RELATION_TYPES_ALL.index(x)).alias("case_index"))
        .sort("case_index")
        .drop("case_index")
    )
    print(df_to_string(df_rel, format_, column_prefixes))


def print_class_table(
    mmref_result_df: pl.DataFrame,
    recall_top_ks: list[int],
    column_prefixes: list[str],
    format_: str = "repr",
) -> None:
    df_class = (
        mmref_result_df.filter(pl.col("rel_type") == "=")
        .group_by("class_name", maintain_order=True)
        .sum()
        .drop(
            [
                "scenario_id",
                "image_id",
                "utterance_id",
                "sid",
                "base_phrase_index",
                "rel_type",
                "instance_id_or_pred_idx",
            ]
        )
    )
    new_columns = [(df_class["precision_pos"] / df_class["precision_total"]).alias("precision")]
    for recall_top_k in recall_top_ks:
        metric_suffix = f"@{recall_top_k}" if recall_top_k >= 0 else ""
        new_columns.append(
            (df_class["recall_pos" + metric_suffix] / df_class["recall_total"]).alias("recall" + metric_suffix)
        )
    df_class = df_class.with_columns(new_columns)
    print(df_to_string(df_class, format_, column_prefixes))


def print_size_table(
    mmref_result_df: pl.DataFrame,
    recall_top_ks: list[int],
    column_prefixes: list[str],
    format_: str = "repr",
) -> None:
    image_size = 1920 * 1080
    mmref_result_df = mmref_result_df.filter(pl.col("rel_type") == "=").with_columns(
        (pl.col("width") * pl.col("height") / image_size).alias("size")
    )
    for min_size, max_size in [(0, 0.005), (0.005, 0.05), (0.05, 1)]:
        df_size = (
            mmref_result_df.filter((pl.col("size") >= min_size) & (pl.col("size") < max_size))
            .sum()
            .drop(
                [
                    "scenario_id",
                    "image_id",
                    "utterance_id",
                    "sid",
                    "base_phrase_index",
                    "rel_type",
                    "instance_id_or_pred_idx",
                ]
            )
        )
        new_columns = [(df_size["precision_pos"] / df_size["precision_total"]).alias("precision")]
        for recall_top_k in recall_top_ks:
            metric_suffix = f"@{recall_top_k}" if recall_top_k >= 0 else ""
            new_columns.append(
                (df_size["recall_pos" + metric_suffix] / df_size["recall_total"]).alias("recall" + metric_suffix)
            )
        df_size = df_size.with_columns(new_columns)
        print((min_size, max_size))
        print(df_to_string(df_size, format_, column_prefixes))


def print_noun_table(
    mmref_result_df: pl.DataFrame,
    recall_top_ks: list[int],
    column_prefixes: list[str],
    format_: str = "repr",
) -> None:
    mmref_result_df = mmref_result_df.filter(pl.col("rel_type") == "=")
    df_class = (
        mmref_result_df.group_by(["pos", "subpos"], maintain_order=True)
        .sum()
        .drop(
            [
                "scenario_id",
                "image_id",
                "utterance_id",
                "sid",
                "base_phrase_index",
                "rel_type",
                "instance_id_or_pred_idx",
            ]
        )
    )
    new_columns = [(df_class["precision_pos"] / df_class["precision_total"]).alias("precision")]
    for recall_top_k in recall_top_ks:
        metric_suffix = f"@{recall_top_k}" if recall_top_k >= 0 else ""
        new_columns.append(
            (df_class["recall_pos" + metric_suffix] / df_class["recall_total"]).alias("recall" + metric_suffix)
        )
    df_class = df_class.with_columns(new_columns)
    print(df_to_string(df_class, format_, column_prefixes))


def print_temporal_table(
    mmref_result_df: pl.DataFrame,
    recall_top_ks: list[int],
    column_prefixes: list[str],
    format_: str = "repr",
) -> None:
    mmref_result_df = mmref_result_df.filter(pl.col("rel_type") == "=")
    df_class = (
        mmref_result_df.group_by(["temporal_location"], maintain_order=True)
        .sum()
        .drop(
            [
                "scenario_id",
                "image_id",
                "utterance_id",
                "sid",
                "base_phrase_index",
                "rel_type",
                "instance_id_or_pred_idx",
            ]
        )
    )
    new_columns = [(df_class["precision_pos"] / df_class["precision_total"]).alias("precision")]
    for recall_top_k in recall_top_ks:
        metric_suffix = f"@{recall_top_k}" if recall_top_k >= 0 else ""
        new_columns.append(
            (df_class["recall_pos" + metric_suffix] / df_class["recall_total"]).alias("recall" + metric_suffix)
        )
    df_class = df_class.with_columns(new_columns)
    print(df_to_string(df_class, format_, column_prefixes))


def df_to_string(table: pl.DataFrame, format_: str, column_prefixes: Optional[list] = None) -> str:
    if column_prefixes is not None:
        columns = [column for column in table.columns if column.startswith(tuple(column_prefixes))]
        table = table.select(*columns)
    if format_ == "repr":
        pl.Config.set_tbl_rows(100)
        pl.Config.set_tbl_cols(16)
        return repr(table)
    elif format_ == "csv":
        return table.write_csv()
    elif format_ == "tsv":
        return table.write_csv(separator="\t")
    elif format_ == "json":
        return table.write_json()
    else:
        raise ValueError(f"Unknown format: {format_}")


if __name__ == "__main__":
    main()
