import argparse
from pathlib import Path
import json

from flickr30k_entities_utils import get_sentence_data, get_sentence_data_ja, get_annotations


def convert(sentence_data: dict, annotation_data: dict, image_id: str, sent_id_from: int = 0, ann_id_from: int = 0):
    sent_id = sent_id_from
    ann_id = ann_id_from
    phrase_id = 0
    images = []
    annotations = []
    # print(annotation_data)
    for idx, sent in enumerate(sentence_data):
        # print(sent)
        sentence: str = sent['sentence']
        words = sentence.split(' ')
        word_to_char_idx = {}
        char_idx = 0
        for word_idx, word in enumerate(words):
            word_to_char_idx[word_idx] = char_idx
            char_idx += len(word) + 1
        char_spans = {}
        for phrase in sent['phrases']:
            if phrase['phrase_id'] not in annotation_data['boxes']:
                continue
            first_char_index: int = word_to_char_idx[phrase['first_word_index']]
            phrase_surf: str = phrase['phrase']
            assert sentence[first_char_index:].startswith(phrase_surf), f"first index mismatch in '{image_id}'"
            char_spans[phrase['phrase_id']] = [[first_char_index, first_char_index + len(phrase_surf)]]

        image = {
            'file_name': f'{image_id}.jpg',
            'height': str(annotation_data['height']),
            'width': str(annotation_data['width']),
            'id': sent_id,
            'caption': sentence,
            'dataset_name': 'flickr',
            'tokens_negative': [[0, len(sentence)]],
            'sentence_id': idx,
            'original_img_id': int(image_id),
            'tokens_positive_eval': list(char_spans.values()),
        }

        for phrase in sent['phrases']:
            for bbox in annotation_data['boxes'].get(phrase['phrase_id'], []):
                x1, y1, x2, y2 = bbox
                annotation = {
                    'area': (x2 - x1) * (y2 - y1),
                    'iscrowd': 0,
                    'image_id': sent_id,
                    'category_id': 1,
                    'id': ann_id,
                    'bbox': [x1 + 1, y1 + 1, x2 - x1, y2 - y1],
                    'tokens_positive': char_spans[phrase['phrase_id']],
                    'phrase_ids': phrase_id,
                }
                annotations.append(annotation)
                ann_id += 1
            phrase_id += 1
        images.append(image)
        sent_id += 1
    return images, annotations


def main():
    parser = argparse.ArgumentParser(description='Convert dataset to COCO format.')
    parser.add_argument('--sentences-dir', '-s', type=str, help='Path to the directory containing captions.')
    parser.add_argument('--annotations-dir', '-a', type=str, help='Path to the directory containing annotations.')
    parser.add_argument('--id-file', '-i', type=str, help='Path to the file containing image ids.')
    parser.add_argument('--sent-id-offset', type=int, default=0, help='Offset for sentence ids.')
    parser.add_argument('--ann-id-offset', type=int, default=0, help='Offset for annotation ids.')
    parser.add_argument('--lang', choices=['en', 'ja'], default='en', help='Language of the captions.')
    args = parser.parse_args()

    sentences_dir = Path(args.sentences_dir)
    annotations_dir = Path(args.annotations_dir)
    image_ids = Path(args.id_file).read_text().splitlines()
    sent_id_offset = args.sent_id_offset
    ann_id_offset = args.ann_id_offset

    all_images = []
    all_annotations = []
    for image_id in image_ids:
        if args.lang == 'en':
            sentence_data = get_sentence_data(sentences_dir / f'{image_id}.txt')
        else:
            sentence_data = get_sentence_data_ja(sentences_dir / f'{image_id}.txt')
        annotation_data = get_annotations(annotations_dir / f'{image_id}.xml')
        images, annotations = convert(
            sentence_data,
            annotation_data,
            image_id,
            sent_id_from=sent_id_offset,
            ann_id_from=ann_id_offset,
        )
        sent_id_offset += len(images)
        ann_id_offset += len(annotations)
        all_images += images
        all_annotations += annotations
    coco_data = {
        'info': [],
        'licenses': [],
        'images': all_images,
        'categories': [{'supercategory': 'object', 'id': 1, 'name': 'object'}],
        'annotations': all_annotations,
    }
    print(json.dumps(coco_data, indent=2, ensure_ascii=False, sort_keys=False))


if __name__ == '__main__':
    main()
