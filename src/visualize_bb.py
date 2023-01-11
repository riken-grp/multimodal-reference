import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile

from prediction_writer import PhraseGroundingResult
from utils.image import ImageAnnotation, ImageTextAnnotation, PhraseAnnotation
from utils.util import DatasetInfo

# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image."""
    for c in range(3):
        image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c] * 255, image[:, :, c])
    return image


def plot_results(
    image: ImageFile, image_annotation: ImageAnnotation, phrase_annotations: list[PhraseAnnotation], export_dir: Path
) -> None:
    plt.figure(figsize=(16, 10))
    np_image = np.array(image)
    ax = plt.gca()
    colors = COLORS * 100

    for bounding_box in image_annotation.bounding_boxes:
        rect = bounding_box.rect
        # score = bounding_box.confidence
        # label = ",".join(word for word, prob in zip(prediction.words, bounding_box.word_probs) if prob >= 0.1)
        labels = []
        for phrase_annotation in phrase_annotations:
            for relation in phrase_annotation.relations:
                if relation.instance_id == bounding_box.instance_id:
                    labels.append(f'{relation.type}{phrase_annotation.text}')
        color = colors.pop()
        ax.add_patch(plt.Rectangle((rect.x1, rect.y1), rect.w, rect.h, fill=False, color=color, linewidth=3))
        ax.text(
            rect.x1,
            rect.y1,
            ', '.join(labels),
            fontsize=15,
            bbox=dict(facecolor=color, alpha=0.8),
            fontname='Hiragino Maru Gothic Pro',
        )

    plt.imshow(np_image)
    plt.axis('off')
    plt.savefig(str(export_dir / f'{image_annotation.image_id}.png'))
    # plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', '-d', type=Path, help='Path to the directory containing the target dataset.')
    parser.add_argument('--image-annotation-file', '-i', type=Path, help='Path to the gold image text annotation file.')
    parser.add_argument('--export-dir', '-e', type=Path, help='Path to the directory where tagged images are exported')
    parser.add_argument('--prediction-file', '-p', type=Path, help='Path to the prediction file.')
    args = parser.parse_args()

    args.export_dir.mkdir(exist_ok=True)
    dataset_info = DatasetInfo.from_json(args.dataset_dir.joinpath('info.json').read_text())
    # gold_document = Document.from_knp(args.gold_knp_dir.joinpath(f'{scenario_id}.knp').read_text())
    image_text_annotation = ImageTextAnnotation.from_json(args.image_annotation_file.read_text())
    prediction = PhraseGroundingResult.from_json(args.prediction_file.read_text())
    utterance_annotations = image_text_annotation.utterances
    image_id_to_annotation = {
        image_annotation.image_id: image_annotation for image_annotation in image_text_annotation.images
    }
    for utterance, utterance_annotation, utterance_result in zip(
        dataset_info.utterances, utterance_annotations, prediction.utterances
    ):
        # base_phrases = [bp for sid in utterance.sids for bp in sid2sentence[sid].base_phrases]
        # assert ''.join(bp.text for bp in base_phrases) == utterance_annotation.text
        for image_id in utterance.image_ids:
            image_annotation = image_id_to_annotation[image_id]
            image = Image.open(args.dataset_dir / f'images/{image_annotation.image_id}.png')
            plot_results(image, image_annotation, utterance_annotation.phrases, args.export_dir)


if __name__ == '__main__':
    main()
