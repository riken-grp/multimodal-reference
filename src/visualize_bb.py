import argparse
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile
from rhoknp import BasePhrase, Document, Sentence

from prediction_writer import PhraseGroundingPrediction, PhrasePrediction
from utils.image import ImageAnnotation, ImageTextAnnotation, PhraseAnnotation
from utils.util import DatasetInfo

# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    # [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],  # light blue
]

GOLD_COLOR = [0.929, 0.694, 0.125]  # yellow


def get_core_expression(unit: Union[BasePhrase]) -> tuple[str, str, str]:
    """A core expression without ancillary words."""
    morphemes = unit.morphemes
    sidx = 0
    for i, morpheme in enumerate(morphemes):
        if morpheme.pos not in ('助詞', '特殊', '判定詞'):
            sidx += i
            break
    eidx = len(morphemes)
    for i, morpheme in enumerate(reversed(morphemes)):
        if morpheme.pos not in ('助詞', '特殊', '判定詞'):
            eidx -= i
            break
    ret = ''.join(m.text for m in morphemes[sidx:eidx])
    if not ret:
        sidx = 0
        eidx = len(morphemes)
    return (
        ''.join(m.text for m in morphemes[:sidx]),
        ''.join(m.text for m in morphemes[sidx:eidx]),
        ''.join(m.text for m in morphemes[eidx:]),
    )


def plot_results(
    image: ImageFile,
    image_annotation: ImageAnnotation,
    phrase_annotations: list[PhraseAnnotation],
    phrase_predictions: list[PhrasePrediction],
    base_phrases: list[BasePhrase],
    export_dir: Path,
    plots: list[str],
    topk: int = -1,
    confidence_threshold: float = 0.0,
) -> None:
    plt.figure(figsize=(16, 10))
    np_image = np.array(image)
    ax = plt.gca()

    if 'pred' in plots:
        draw_prediction(ax, base_phrases, confidence_threshold, image_annotation, phrase_predictions, topk)

    if 'gold' in plots:
        draw_annotation(ax, base_phrases, image_annotation, phrase_annotations)

    plt.imshow(np_image)
    plt.axis('off')
    plt.savefig(str(export_dir / f'{image_annotation.image_id}.png'))
    # plt.show()


def draw_annotation(ax, base_phrases, image_annotation, phrase_annotations):
    for bounding_box in image_annotation.bounding_boxes:
        rect = bounding_box.rect
        labels = []
        for phrase_annotation in phrase_annotations:
            base_phrase = next(filter(lambda bp: bp.text == phrase_annotation.text, base_phrases))
            for relation in phrase_annotation.relations:
                if relation.instance_id == bounding_box.instance_id:
                    labels.append(f'{relation.type}_{get_core_expression(base_phrase)[1]}')
        if not labels:
            continue
        # color = colors.pop()
        ax.add_patch(plt.Rectangle((rect.x1, rect.y1), rect.w, rect.h, fill=False, color=GOLD_COLOR, linewidth=3))
        ax.text(
            rect.x1,
            rect.y1,
            ', '.join(labels),
            fontsize=24,
            bbox=dict(facecolor=GOLD_COLOR, alpha=0.8),
            fontname='Hiragino Maru Gothic Pro',
        )


def draw_prediction(ax, base_phrases, confidence_threshold, image_annotation, phrase_predictions, topk):
    colors = COLORS * 100
    for phrase_prediction in phrase_predictions:
        relation_types: set[str] = {relation.type for relation in phrase_prediction.relations}
        for relation_type in relation_types:
            relations = [
                r
                for r in phrase_prediction.relations
                if r.image_id == image_annotation.image_id
                and r.type == relation_type
                and r.bounding_box.confidence >= confidence_threshold
            ]
            sorted_relations = sorted(relations, key=lambda r: r.bounding_box.confidence, reverse=True)
            for pred_relation in sorted_relations[:topk] if topk >= 0 else sorted_relations:
                pred_bounding_box = pred_relation.bounding_box
                rect = pred_bounding_box.rect
                base_phrase = next(filter(lambda bp: bp.text == phrase_prediction.text, base_phrases))
                label = '{type}_{text}: {score:0.2f}'.format(
                    type=pred_relation.type,
                    text=get_core_expression(base_phrase)[1],
                    score=pred_bounding_box.confidence,
                )
                color = colors.pop()
                ax.add_patch(plt.Rectangle((rect.x1, rect.y1), rect.w, rect.h, fill=False, color=color, linewidth=3))
                ax.text(
                    rect.x1,
                    rect.y1,
                    label,
                    fontsize=24,
                    bbox=dict(facecolor=color, alpha=0.8),
                    fontname='Hiragino Maru Gothic Pro',
                )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', '-d', type=Path, help='Path to the directory containing the target dataset.')
    parser.add_argument('--gold-knp-dir', '-k', type=Path, help='Path to the gold KNP directory.')
    parser.add_argument('--image-annotation-dir', '-i', type=Path, help='Path to the gold image text annotation file.')
    parser.add_argument('--export-dir', '-e', type=Path, help='Path to the directory where tagged images are exported')
    parser.add_argument('--prediction-dir', '-p', type=Path, help='Path to the prediction file.')
    parser.add_argument('--scenario-ids', '--ids', type=str, nargs='*', help='List of scenario ids.')
    parser.add_argument('--plots', type=str, nargs='*', choices=["gold", "pred"], help='Plotting target.')
    return parser.parse_args()


def main():
    args = parse_args()
    for scenario_id in args.scenario_ids:
        export_dir = args.export_dir / scenario_id
        export_dir.mkdir(parents=True, exist_ok=True)
        dataset_info = DatasetInfo.from_json(args.dataset_dir.joinpath(f'{scenario_id}/info.json').read_text())
        image_dir = args.dataset_dir / scenario_id / 'images'
        gold_document = Document.from_knp(args.gold_knp_dir.joinpath(f'{scenario_id}.knp').read_text())
        image_text_annotation = ImageTextAnnotation.from_json(
            args.image_annotation_dir.joinpath(f'{scenario_id}.json').read_text()
        )
        prediction = PhraseGroundingPrediction.from_json(
            args.prediction_dir.joinpath(f'{scenario_id}.json').read_text()
        )
        visualize(export_dir, dataset_info, gold_document, image_dir, image_text_annotation, prediction, args.plots)


def visualize(
    export_dir: Path,
    dataset_info: DatasetInfo,
    gold_document: Document,
    image_dir: Path,
    image_text_annotation: ImageTextAnnotation,
    prediction: PhraseGroundingPrediction,
    plots: list[str],
) -> None:
    utterance_annotations = image_text_annotation.utterances
    image_id_to_annotation = {
        image_annotation.image_id: image_annotation for image_annotation in image_text_annotation.images
    }
    sid2sentence: dict[str, Sentence] = {sentence.sid: sentence for sentence in gold_document.sentences}
    for utterance, utterance_annotation, utterance_prediction in zip(
        dataset_info.utterances, utterance_annotations, prediction.utterances
    ):
        base_phrases = [bp for sid in utterance.sids for bp in sid2sentence[sid].base_phrases]
        assert ''.join(bp.text for bp in base_phrases) == utterance_annotation.text
        for image_id in utterance.image_ids:
            image_annotation = image_id_to_annotation[image_id]
            image = Image.open(image_dir / f'{image_annotation.image_id}.png')
            plot_results(
                image,
                image_annotation,
                utterance_annotation.phrases,
                utterance_prediction.phrases,
                base_phrases,
                export_dir,
                confidence_threshold=0.9,
                plots=plots,
            )


if __name__ == '__main__':
    main()
