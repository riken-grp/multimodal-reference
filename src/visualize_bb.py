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
    topk: int = 1,
) -> None:
    plt.figure(figsize=(16, 10))
    np_image = np.array(image)
    ax = plt.gca()
    colors = COLORS * 100

    for phrase_prediction in phrase_predictions:
        bounding_boxes = sorted(phrase_prediction.bounding_boxes, key=lambda bb: bb.confidence, reverse=True)
        for pred_bounding_box in bounding_boxes[:topk]:
            rect = pred_bounding_box.rect
            score = pred_bounding_box.confidence
            # if score < 0.8:
            #     continue
            base_phrase = next(filter(lambda bp: bp.text == phrase_prediction.text, base_phrases))
            label = f'=_{get_core_expression(base_phrase)[1]}: {score:0.2f}'
            labels = [label]
            color = colors.pop()
            ax.add_patch(plt.Rectangle((rect.x1, rect.y1), rect.w, rect.h, fill=False, color=color, linewidth=3))
            ax.text(
                rect.x1,
                rect.y1,
                ', '.join(labels),
                fontsize=24,
                bbox=dict(facecolor=color, alpha=0.8),
                fontname='Hiragino Maru Gothic Pro',
            )

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

    plt.imshow(np_image)
    plt.axis('off')
    plt.savefig(str(export_dir / f'{image_annotation.image_id}.png'))
    # plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', '-d', type=Path, help='Path to the directory containing the target dataset.')
    parser.add_argument('--gold-knp-dir', '-k', type=Path, help='Path to the gold KNP directory.')
    parser.add_argument('--image-annotation-dir', '-i', type=Path, help='Path to the gold image text annotation file.')
    parser.add_argument('--export-dir', '-e', type=Path, help='Path to the directory where tagged images are exported')
    parser.add_argument('--prediction-dir', '-p', type=Path, help='Path to the prediction file.')
    parser.add_argument('--scenario-id', '--id', type=Path, help='Scenario id.')
    args = parser.parse_args()

    scenario_id = args.scenario_id
    export_dir = args.export_dir / scenario_id
    export_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = DatasetInfo.from_json(args.dataset_dir.joinpath(f'{scenario_id}/info.json').read_text())
    gold_document = Document.from_knp(args.gold_knp_dir.joinpath(f'{scenario_id}.knp').read_text())
    image_text_annotation = ImageTextAnnotation.from_json(
        args.image_annotation_dir.joinpath(f'{scenario_id}.json').read_text()
    )
    prediction = PhraseGroundingPrediction.from_json(args.prediction_dir.joinpath(f'{scenario_id}.json').read_text())
    dataset_dir = args.dataset_dir / scenario_id
    utterance_annotations = image_text_annotation.utterances
    image_id_to_annotation = {
        image_annotation.image_id: image_annotation for image_annotation in image_text_annotation.images
    }
    sid2sentence: dict[str, Sentence] = {sentence.sid: sentence for sentence in gold_document.sentences}
    for utterance, utterance_annotation, utterance_result in zip(
        dataset_info.utterances, utterance_annotations, prediction.utterances
    ):
        base_phrases = [bp for sid in utterance.sids for bp in sid2sentence[sid].base_phrases]
        base_phrase_keys = [(bp.sentence.sid, bp.index) for bp in base_phrases]
        assert ''.join(bp.text for bp in base_phrases) == utterance_annotation.text
        for image_id in utterance.image_ids:
            image_annotation = image_id_to_annotation[image_id]
            image = Image.open(dataset_dir / f'images/{image_annotation.image_id}.png')
            pred_phrases: list[PhrasePrediction] = list(
                filter(
                    lambda p: p.image.id == image_id and (p.sid, p.index) in base_phrase_keys,
                    utterance_result.phrases,
                )
            )
            plot_results(image, image_annotation, utterance_annotation.phrases, pred_phrases, base_phrases, export_dir)


if __name__ == '__main__':
    main()
