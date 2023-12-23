import argparse
import math
from dataclasses import dataclass
from itertools import cycle, repeat
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox
from PIL import Image, ImageFile
from rhoknp import BasePhrase, Document, Sentence

from utils.annotation import ImageAnnotation, ImageTextAnnotation, PhraseAnnotation
from utils.constants import RELATION_TYPES_ALL
from utils.prediction import PhraseGroundingPrediction, PhrasePrediction
from utils.util import DatasetInfo, Rectangle

# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],  # blue
    [0.850, 0.325, 0.098],  # orange
    # [0.929, 0.694, 0.125],  # yellow
    [0.494, 0.184, 0.556],  # purple
    [0.466, 0.674, 0.188],  # green
    [0.301, 0.745, 0.933],  # light blue
]

GOLD_COLOR = [0.929, 0.694, 0.125]  # yellow


@dataclass
class LabeledRectangle:
    rect: Rectangle
    color: list[float]
    label: str


def get_core_expression(unit: Union[BasePhrase]) -> tuple[str, str, str]:
    """A core expression without ancillary words."""
    morphemes = unit.morphemes
    sidx = 0
    for i, morpheme in enumerate(morphemes):
        if morpheme.pos not in ("助詞", "特殊", "判定詞"):
            sidx += i
            break
    eidx = len(morphemes)
    for i, morpheme in enumerate(reversed(morphemes)):
        if morpheme.pos not in ("助詞", "特殊", "判定詞"):
            eidx -= i
            break
    ret = "".join(m.text for m in morphemes[sidx:eidx])
    if not ret:
        sidx = 0
        eidx = len(morphemes)
    return (
        "".join(m.text for m in morphemes[:sidx]),
        "".join(m.text for m in morphemes[sidx:eidx]),
        "".join(m.text for m in morphemes[eidx:]),
    )


def plot_results(
    image: ImageFile,
    image_annotation: ImageAnnotation,
    phrase_annotations: list[PhraseAnnotation],
    phrase_predictions: list[PhrasePrediction],
    base_phrases: list[BasePhrase],
    export_dir: Path,
    plots: list[str],
    relation_types: set[str],
    confidence_threshold: float = 0.0,
    topk: int = -1,
    class_names: Optional[set[str]] = None,
) -> None:
    fig = plt.figure(figsize=(16, 10))
    np_image = np.array(image)
    ax = fig.add_subplot(111)

    labeled_rectangles: list[LabeledRectangle] = []

    if "pred" in plots:
        labeled_rectangles += draw_prediction(
            base_phrases, phrase_predictions, image_annotation.image_id, confidence_threshold, topk, relation_types
        )

    if "gold" in plots:
        labeled_rectangles += draw_annotation(
            base_phrases, phrase_annotations, image_annotation, relation_types, class_names
        )

    drawn_bbs: set[Bbox] = set()
    for labeled_rectangle in labeled_rectangles:
        rect = labeled_rectangle.rect
        ax.add_patch(
            plt.Rectangle((rect.x1, rect.y1), rect.w, rect.h, fill=False, color=labeled_rectangle.color, linewidth=3)
        )
        text_box = ax.text(
            rect.x1,
            rect.y1,
            labeled_rectangle.label,
            fontsize=24,
            bbox=dict(facecolor=labeled_rectangle.color, alpha=0.8),
            fontname="Hiragino Maru Gothic Pro",
        )
        fig.canvas.draw()
        bbox_window = text_box.get_window_extent()
        bbox = bbox_window.transformed(ax.transData.inverted())
        bbox = Bbox.from_bounds(bbox.x0, bbox.y0, bbox_window.width, bbox_window.height)

        count = 0
        stride = 10
        while any(Bbox.intersection(bb, bbox) is not None for bb in drawn_bbs) and count < 50:
            text_box.set_y(text_box.get_position()[1] - stride)
            bbox = Bbox.from_bounds(bbox.x0, bbox.y0 - stride, bbox.width, bbox.height)
            count += 1
        drawn_bbs.add(bbox)

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
                    labels.append(f"{relation.type}_{get_core_expression(base_phrase)[1]}")
        if not labels:
            continue
        ret.append(
            LabeledRectangle(
                rect=rect,
                color=GOLD_COLOR,
                label=", ".join(labels),
            )
        )
    return ret


def draw_prediction(
    base_phrases: list[BasePhrase],
    phrase_predictions: list[PhrasePrediction],
    image_id: str,
    confidence_threshold: float,
    topk: int,
    relation_types: set[str],
) -> list[LabeledRectangle]:
    colors = cycle(COLORS)
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
                label = "{type}_{text}: {score:0.2f}".format(
                    type=pred_relation.type,
                    text=get_core_expression(base_phrase)[1],
                    score=pred_bounding_box.confidence,
                )
                ret.append(LabeledRectangle(rect=rect, color=next(colors), label=label))
    return ret


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str, help="Experiment name (directory name under --export-dir)")
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
    parser.add_argument("--class-names", type=str, nargs="*", default=None, help="Class names to visualize.")
    return parser.parse_args()


def main():
    args = parse_args()
    for scenario_id in args.scenario_ids:
        export_dir = args.export_dir / args.exp_name / scenario_id
        export_dir.mkdir(parents=True, exist_ok=True)
        dataset_info = DatasetInfo.from_json(args.dataset_dir.joinpath(f"{scenario_id}/info.json").read_text())
        image_dir = args.dataset_dir / scenario_id / "images"
        gold_document = Document.from_knp(args.gold_knp_dir.joinpath(f"{scenario_id}.knp").read_text())
        image_text_annotation = ImageTextAnnotation.from_json(
            args.image_annotation_dir.joinpath(f"{scenario_id}.json").read_text()
        )
        prediction_file = args.prediction_dir / args.exp_name / f"{scenario_id}.json"
        if prediction_file.exists():
            prediction = PhraseGroundingPrediction.from_json(prediction_file.read_text())
        else:
            print(f"Warning: {prediction_file} does not exist.")
            prediction = None

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
                    confidence_threshold=args.confidence_threshold,
                    topk=args.topk,
                    class_names=set(args.class_names) if args.class_names is not None else None,
                )


if __name__ == "__main__":
    main()
