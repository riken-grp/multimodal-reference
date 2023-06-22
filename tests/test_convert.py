import json
from pathlib import Path

from scripts.convert_dataset import convert  # noqa: E402
from scripts.flickr30k_entities_utils import get_annotations, get_sentence_data  # noqa: E402

here = Path(__file__).resolve().parent
data_dir = here / "data"


def test_flickr30k_coco():
    image_ids = data_dir.joinpath("sample_ids.txt").read_text().splitlines()

    all_images = []
    all_annotations = []
    for image_id in image_ids:
        sentence_data = get_sentence_data(data_dir / "flickr30k_sentences" / f"{image_id}.txt")
        annotation_data = get_annotations(data_dir / "flickr30k_annotations" / f"{image_id}.xml")
        images, annotations = convert(sentence_data, annotation_data, image_id, sent_id_from=153901, ann_id_from=662765)
        all_images.extend(images)
        all_annotations.extend(annotations)
    coco_data = {
        "info": [],
        "licenses": [],
        "images": all_images,
        "categories": [{"supercategory": "object", "id": 1, "name": "object"}],
        "annotations": all_annotations,
    }
    expected = json.loads(data_dir.joinpath("flickr30k_coco.json").read_text())
    assert coco_data["info"] == expected["info"]
    assert coco_data["licenses"] == expected["licenses"]
    assert coco_data["images"] == expected["images"]
    assert coco_data["annotations"] == expected["annotations"]
    assert coco_data["categories"] == expected["categories"]
