import argparse
import itertools
import math
import random
import sys
from dataclasses import dataclass
from itertools import cycle, repeat
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.transforms import Bbox
from PIL import Image, ImageFile
from rhoknp import BasePhrase, Document, Sentence

from utils.annotation import ImageAnnotation, ImageTextAnnotation, PhraseAnnotation
from utils.constants import RELATION_TYPES_ALL
from utils.prediction import PhraseGroundingPrediction, PhrasePrediction
from utils.util import DatasetInfo, IdMapper, Rectangle, get_core_expression

GOLD_COLOR = (1.0, 1.0, 1.0)  # white


@dataclass
class LabeledRectangle:
    rect: Rectangle
    color: tuple[float, float, float]
    label: str


def box_iou(box1: Bbox, box2: Bbox) -> float:
    intersection = Bbox.intersection(box1, box2)
    if intersection is None:
        return 0
    intersection_area = intersection.width * intersection.height
    box1_area = box1.width * box1.height
    box2_area = box2.width * box2.height
    union_area = box1_area + box2_area - intersection_area
    if union_area == 0:
        return 0
    return intersection_area / union_area


def box_io1(box1: Bbox, box2: Bbox) -> float:
    intersection = Bbox.intersection(box1, box2)
    if intersection is None:
        return 0
    intersection_area = intersection.width * intersection.height
    box1_area = box1.width * box1.height
    if box1_area == 0:
        return 0
    return intersection_area / box1_area


def plot_results(
    image: ImageFile,
    image_annotation: ImageAnnotation,
    phrase_annotations: list[PhraseAnnotation],
    phrase_predictions: list[PhrasePrediction],
    base_phrases: list[BasePhrase],
    export_dir: Path,
    plots: list[str],
    relation_types: set[str],
    id_mapper: Optional[IdMapper],
    confidence_threshold: float = 0.0,
    topk: int = -1,
    class_names: Optional[set[str]] = None,
    phrases: Optional[set[str]] = None,
    hide_utterance: bool = False,
) -> None:
    fig = plt.figure(figsize=(16, 10))
    np_image = np.array(image)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.01, right=0.8, bottom=0.01, top=0.8)

    labeled_rectangles: list[LabeledRectangle] = []

    if "pred" in plots:
        labeled_rectangles += draw_prediction(
            base_phrases,
            phrase_predictions,
            image_annotation.image_id,
            confidence_threshold,
            topk,
            relation_types,
            phrases,
        )

    if "gold" in plots:
        labeled_rectangles += draw_annotation(
            base_phrases, phrase_annotations, image_annotation, relation_types, class_names, phrases, id_mapper
        )

    drawn_object_bbs: list[Bbox] = []
    for labeled_rectangle in labeled_rectangles:
        rect = labeled_rectangle.rect
        kwargs = dict(fill=False, color=labeled_rectangle.color, linewidth=3)
        if labeled_rectangle.color == GOLD_COLOR:
            kwargs["linestyle"] = "--"
        object_rectangle = plt.Rectangle((rect.x1, rect.y1), rect.w, rect.h, **kwargs)
        ax.add_patch(object_rectangle)
        drawn_object_bbs.append(object_rectangle.get_bbox())

    drawn_label_bbs: set[Bbox] = set()
    for labeled_rectangle, object_bbox in zip(labeled_rectangles, drawn_object_bbs):
        rect = labeled_rectangle.rect
        text_box = ax.text(
            rect.x1,
            rect.y1 - 25,
            labeled_rectangle.label,
            fontsize=24,
            bbox=dict(facecolor=labeled_rectangle.color, alpha=0.8),
            fontname="Hiragino Maru Gothic Pro",
        )
        fig.canvas.draw()
        label_bbox_window = text_box.get_window_extent()
        label_bbox = Bbox.from_bounds(
            rect.x1, rect.y1 - 70, label_bbox_window.width * 0.78, 70 * len(labeled_rectangle.label.splitlines())
        )

        stride = 10
        count = 0
        while any(box_iou(bb, label_bbox) >= 0.01 for bb in drawn_label_bbs) and count < 50:
            text_box.set_y(text_box.get_position()[1] - stride)
            label_bbox = Bbox.from_bounds(label_bbox.x0, label_bbox.y0 - stride, label_bbox.width, label_bbox.height)
            count += 1

        count = 0
        while any(box_io1(bb, label_bbox) >= 0.1 for bb in drawn_object_bbs if bb != object_bbox) and count < 50:
            text_box.set_x(text_box.get_position()[0] - stride)
            label_bbox = Bbox.from_bounds(label_bbox.x0 - stride, label_bbox.y0, label_bbox.width, label_bbox.height)
            count += 1

        drawn_label_bbs.add(label_bbox)

    if not hide_utterance:
        ax.text(
            -10,
            -50,
            "".join(bp.text for bp in base_phrases),
            fontsize=24,
            bbox=dict(facecolor=(1.0, 1.0, 1.0), alpha=0.8),
            fontname="Hiragino Maru Gothic Pro",
        )

    plt.imshow(np_image)
    plt.axis("off")
    fig.savefig(str(export_dir / f"{image_annotation.image_id}.png"))
    # plt.show()
    plt.close(fig)


def draw_annotation(
    base_phrases: list[BasePhrase],
    phrase_annotations: list[PhraseAnnotation],
    image_annotation: ImageAnnotation,
    relation_types: set[str],
    class_names: Optional[set[str]],
    phrases: Optional[set[str]],
    id_mapper: Optional[IdMapper],
) -> list[LabeledRectangle]:
    ret = []
    for bounding_box in image_annotation.bounding_boxes:
        if class_names is not None and bounding_box.class_name not in class_names:
            continue
        rect = bounding_box.rect
        labels = []
        for phrase_annotation in phrase_annotations:
            base_phrase = next(filter(lambda bp: bp.text == phrase_annotation.text, base_phrases))
            for relation in phrase_annotation.relations:
                if relation.type not in relation_types:
                    continue
                if relation.instance_id == bounding_box.instance_id:
                    core_expression = get_core_expression(base_phrase)[1]
                    if phrases is not None and core_expression not in phrases:
                        continue
                    labels.append(f"{relation.type}_{core_expression}")
        if not labels:
            continue
        label = ", ".join(labels)
        instance_id = bounding_box.instance_id
        if id_mapper is not None:
            instance_id = str(id_mapper.map(instance_id))
        label += f": {bounding_box.class_name}_{instance_id}"
        ret.append(LabeledRectangle(rect=rect, color=GOLD_COLOR, label=label))
    return ret


def draw_prediction(
    base_phrases: list[BasePhrase],
    phrase_predictions: list[PhrasePrediction],
    image_id: str,
    confidence_threshold: float,
    topk: int,
    relation_types: set[str],
    phrases: Optional[set[str]],
) -> list[LabeledRectangle]:
    base_colors: list[tuple] = list(colormaps["tab20"].colors)
    random.seed(0)
    random.shuffle(base_colors)
    colors = cycle(base_colors)
    ret = []
    for phrase_prediction in phrase_predictions:
        target_relation_types: set[str] = {relation.type for relation in phrase_prediction.relations} & relation_types
        for relation_type in target_relation_types:
            relations = [
                r
                for r in phrase_prediction.relations
                if r.image_id == image_id
                and r.type == relation_type
                and r.bounding_box.confidence >= confidence_threshold
            ]
            sorted_relations = sorted(relations, key=lambda r: r.bounding_box.confidence, reverse=True)
            for pred_relation in sorted_relations[:topk] if topk >= 0 else sorted_relations:
                pred_bounding_box = pred_relation.bounding_box
                rect = pred_bounding_box.rect
                base_phrase = next(filter(lambda bp: bp.text == phrase_prediction.text, base_phrases))
                core_expression = get_core_expression(base_phrase)[1]
                if phrases is not None and core_expression not in phrases:
                    continue
                label = f"{pred_relation.type}_{core_expression}: {pred_bounding_box.confidence:0.2f}"
                ret.append(LabeledRectangle(rect=rect, color=next(colors)[:3], label=label))
    return ret


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_names", type=str, nargs="+", help="Experiment name.")
    parser.add_argument(
        "--dataset-dir",
        "-d",
        type=Path,
        default="data/dataset",
        help="Path to the directory containing the target dataset.",
    )
    parser.add_argument("--gold-knp-dir", "-k", type=Path, default="data/knp", help="Path to the gold KNP directory.")
    parser.add_argument(
        "--image-annotation-dir",
        "-i",
        type=Path,
        default="data/image_text_annotation",
        help="Path to the gold image text annotation file.",
    )
    parser.add_argument(
        "--export-dir",
        "-e",
        type=Path,
        default="data/bb",
        help="Path to the directory where tagged images are exported",
    )
    parser.add_argument(
        "--prediction-dir", "-p", type=Path, default="result/mmref", help="Path to the prediction file."
    )
    parser.add_argument("--scenario-ids", "--ids", type=str, nargs="*", help="List of scenario ids.")
    parser.add_argument(
        "--plots", type=str, nargs="*", choices=["gold", "pred"], default=["gold", "pred"], help="Plotting target."
    )
    parser.add_argument(
        "--relation-types",
        "--rels",
        type=str,
        nargs="*",
        choices=RELATION_TYPES_ALL,
        default=RELATION_TYPES_ALL,
        help="Relation types to plot.",
    )
    parser.add_argument(
        "--confidence-threshold",
        "--th",
        type=float,
        default=0.0,
        help="Confidence threshold for predicted bounding boxes.",
    )
    parser.add_argument("--topk", type=int, default=-1, help="Visualizing top-k predictions.")
    parser.add_argument(
        "--class-names", "--classes", type=str, nargs="*", default=None, help="Class names to visualize."
    )
    parser.add_argument("--phrases", type=str, nargs="*", default=None, help="Phrases to visualize.")
    parser.add_argument("--int-instance-id", action="store_true", help="Use integer instance id.")
    parser.add_argument("--hide-utterance", action="store_true", help="Hide utterance text.")
    return parser.parse_args()


def main():
    args = parse_args()
    for exp_name, scenario_id in itertools.product(args.exp_names, args.scenario_ids):
        export_dir: Path = args.export_dir / f"top{args.topk}_th{args.confidence_threshold}" / exp_name / scenario_id
        export_dir.mkdir(parents=True, exist_ok=True)
        dataset_info = DatasetInfo.from_json(args.dataset_dir.joinpath(f"{scenario_id}/info.json").read_text())
        image_dir: Path = args.dataset_dir / scenario_id / "images"
        gold_document = Document.from_knp(args.gold_knp_dir.joinpath(f"{scenario_id}.knp").read_text())
        image_text_annotation = ImageTextAnnotation.from_json(
            args.image_annotation_dir.joinpath(f"{scenario_id}.json").read_text()
        )
        prediction_file = args.prediction_dir / exp_name / f"{scenario_id}.json"
        if prediction_file.exists():
            prediction = PhraseGroundingPrediction.from_json(prediction_file.read_text())
        else:
            print(f"Warning: {prediction_file} does not exist.", file=sys.stderr)
            prediction = None

        id_mapper = IdMapper() if args.int_instance_id else None

        utterance_annotations = image_text_annotation.utterances
        image_id_to_annotation = {
            image_annotation.image_id: image_annotation for image_annotation in image_text_annotation.images
        }
        sid2sentence: dict[str, Sentence] = {sentence.sid: sentence for sentence in gold_document.sentences}
        all_image_ids = [image.id for image in dataset_info.images]
        assert len(dataset_info.utterances) == len(utterance_annotations)
        for idx, (utterance, utterance_annotation, utterance_prediction) in enumerate(
            zip(dataset_info.utterances, utterance_annotations, prediction.utterances if prediction else repeat(None))
        ):
            base_phrases = [bp for sid in utterance.sids for bp in sid2sentence[sid].base_phrases]
            assert "".join(bp.text for bp in base_phrases) == utterance_annotation.text
            start_index = math.ceil(utterance.start / 1000)
            if idx + 1 < len(dataset_info.utterances):
                next_utterance = dataset_info.utterances[idx + 1]
                end_index = math.ceil(next_utterance.start / 1000)
            else:
                end_index = len(all_image_ids)
            for image_id in all_image_ids[start_index:end_index]:
                plot_results(
                    image=Image.open(image_dir / f"{image_id}.png"),
                    image_annotation=image_id_to_annotation[image_id],
                    phrase_annotations=utterance_annotation.phrases,
                    phrase_predictions=utterance_prediction.phrases if utterance_prediction else [],
                    base_phrases=base_phrases,
                    export_dir=export_dir,
                    plots=args.plots,
                    relation_types=set(args.relation_types),
                    id_mapper=id_mapper,
                    confidence_threshold=args.confidence_threshold,
                    topk=args.topk,
                    class_names=set(args.class_names) if args.class_names is not None else None,
                    phrases=set(args.phrases) if args.phrases is not None else None,
                    hide_utterance=args.hide_utterance,
                )


if __name__ == "__main__":
    main()
