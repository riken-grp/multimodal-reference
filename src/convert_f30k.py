import argparse
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from rhoknp import KWJA, Document, Jumanpp
from rhoknp import Sentence as KNPSentence

from utils.annotation import (
    BoundingBox,
    ImageAnnotation,
    ImageTextAnnotation,
    UtteranceAnnotation,
)
from utils.util import CamelCaseDataClassJsonMixin, DatasetInfo, ImageInfo, Rectangle, UtteranceInfo


@dataclass(frozen=True)
class BndBox(CamelCaseDataClassJsonMixin):
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass(frozen=True)
class Obj(CamelCaseDataClassJsonMixin):
    name: str
    bndbox: BndBox


@dataclass(frozen=True)
class Size(CamelCaseDataClassJsonMixin):
    width: int
    height: int
    depth: int


@dataclass(frozen=True)
class Annotation(CamelCaseDataClassJsonMixin):
    filename: str
    size: Size
    objects: list[Obj]

    @classmethod
    def from_xml(cls, xml: ET.Element) -> "Annotation":
        filename = xml.find("filename").text  # type: ignore
        assert filename is not None
        size = xml.find("size")
        assert size is not None
        width = int(size.find("width").text)  # type: ignore
        height = int(size.find("height").text)  # type: ignore
        depth = int(size.find("depth").text)  # type: ignore
        objects = []
        for obj in xml.findall("object"):
            name = obj.find("name").text  # type: ignore
            if (bndbox := obj.find("bndbox")) is not None:
                xmin = int(bndbox.find("xmin").text)  # type: ignore
                ymin = int(bndbox.find("ymin").text)  # type: ignore
                xmax = int(bndbox.find("xmax").text)  # type: ignore
                ymax = int(bndbox.find("ymax").text)  # type: ignore
                objects.append(Obj(name=name, bndbox=BndBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)))  # type: ignore
            else:
                assert obj.find("nobndbox") is not None
        return cls(filename=filename, size=Size(width=width, height=height, depth=depth), objects=objects)


@dataclass(frozen=True)
class Phrase:
    tag_idx: int
    text: str
    span: tuple[int, int]
    phrase_id: int
    phrase_type: str

    def to_string(self) -> str:
        return f"[/EN#{self.phrase_id}/{self.phrase_type} {self.text}]"


@dataclass(frozen=True)
class Sentence:
    text: str
    phrases: list[Phrase]

    def to_string(self) -> str:
        # 3:[/EN#549/people ほかの男]が立って4:[/EN#551/other ロープ]を握っている間に、1:[/EN#547/people ７人のクライマー]が2:[/EN#548/bodyparts 岩壁]を登っている。
        output_string = ""
        cursor = 0
        for phrase in self.phrases:
            output_string += self.text[cursor : phrase.span[0]]
            output_string += f"{phrase.tag_idx}:" + phrase.to_string()
            cursor = phrase.span[1]
        output_string += self.text[cursor:]
        return output_string

    @classmethod
    def from_string(cls, sentence_string: str) -> "Sentence":
        # 3:[/EN#549/people ほかの男]が立って4:[/EN#551/other ロープ]を握っている間に、1:[/EN#547/people ７人のクライマー]が2:[/EN#548/bodyparts 岩壁]を登っている。
        tag_pat = re.compile(r"(?P<idx>[0-9]+):\[/EN#(?P<id>[0-9]+)(/(?P<type>[A-Za-z_\-()/]+))+ (?P<words>[^]]+)]")
        chunks: list[Union[str, dict[str, Any]]] = []
        sidx = 0
        matches: list[re.Match] = list(re.finditer(tag_pat, sentence_string))
        for match in matches:
            # chunk 前を追加
            if sidx < match.start():
                text = sentence_string[sidx : match.start()]
                chunks.append(text)
            # match の中身を追加
            chunks.append(
                {
                    "tag_idx": match.group("idx"),
                    "phrase": match.group("words"),
                    "phrase_id": match.group("id"),
                    "phrase_type": match.group("type"),
                }
            )
            sidx = match.end()
        # chunk 後を追加
        if sidx < len(sentence_string):
            chunks.append(sentence_string[sidx:])
        sentence = ""
        phrases = []
        char_idx = 0
        for chunk in chunks:
            if isinstance(chunk, str):
                sentence += chunk
                char_idx += len(chunk)
            else:
                chunk["first_char_index"] = char_idx
                sentence += chunk["phrase"]
                char_idx += len(chunk["phrase"])
                phrases.append(
                    Phrase(
                        tag_idx=int(chunk["tag_idx"]),
                        text=chunk["phrase"],
                        span=(chunk["first_char_index"], chunk["first_char_index"] + len(chunk["phrase"])),
                        phrase_id=int(chunk["phrase_id"]),
                        phrase_type=chunk["phrase_type"],
                    )
                )
        assert "EN" not in sentence
        return cls(sentence.strip(), phrases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-file", type=Path, help="Path to Flickr30k ID file.")
    parser.add_argument("--flickr-image-dir", type=str, help="Path to flickr image directory.")
    parser.add_argument("--flickr-annotations-dir", type=str, help="Path to flickr Annotations directory.")
    parser.add_argument("--flickr-sentences-dir", type=str, help="Path to flickr Sentences directory.")
    parser.add_argument("--dataset-dir", "-d", type=str, help="Path to dataset directory.")
    parser.add_argument("--knp-dir", "-k", type=Path, help="Path to knp directory.")
    parser.add_argument("--annotation-dir", "-a", type=str, help="Path to annotation directory.")
    args = parser.parse_args()

    ids = args.id_file.read_text().splitlines()
    flickr_image_dir = Path(args.flickr_image_dir)
    flickr_annotations_dir = Path(args.flickr_annotations_dir)
    flickr_sentences_dir = Path(args.flickr_sentences_dir)
    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(exist_ok=True)
    knp_dir: Path = args.knp_dir
    knp_dir.mkdir(exist_ok=True)
    annotation_dir = Path(args.annotation_dir)
    annotation_dir.mkdir(exist_ok=True)

    for flickr_image_id in ids:
        flickr_annotation_file = flickr_annotations_dir / f"{flickr_image_id}.xml"
        flickr_sentences_file = flickr_sentences_dir / f"{flickr_image_id}.txt"
        flickr_annotation = Annotation.from_xml(ET.parse(flickr_annotation_file).getroot())
        flickr_sentences = []
        for flickr_sentence in flickr_sentences_file.read_text().splitlines():
            sentence = Sentence.from_string(flickr_sentence)
            assert flickr_sentence == sentence.to_string()
            flickr_sentences.append(sentence)
        convert_flickr(
            f"{int(flickr_image_id):010d}",
            flickr_annotation,
            flickr_sentences,
            flickr_image_dir,
            dataset_dir,
            annotation_dir,
            knp_dir,
        )


def convert_flickr(
    flickr_image_id: str,
    flickr_annotation: Annotation,
    flickr_sentences: list[Sentence],
    flickr_image_dir: Path,
    dataset_dir: Path,
    annotation_dir: Path,
    knp_dir: Path,
) -> None:
    for idx, flickr_sentence in enumerate(flickr_sentences):
        scenario_id = f"{flickr_image_id}{idx:02d}"
        image_dir: Path = dataset_dir / scenario_id / "images"
        dataset_info = DatasetInfo(
            scenario_id=scenario_id,
            utterances=[
                UtteranceInfo(
                    text=flickr_sentence.text,
                    sids=[f"{flickr_image_id}{idx:02d}"],
                    start=0,
                    end=1,
                    duration=1,
                    speaker="",
                    image_ids=[flickr_image_id],
                )
            ],
            images=[
                ImageInfo(
                    id=flickr_image_id,
                    path=f"images/{flickr_image_id}.jpg",
                    time=0,
                )
            ],
        )
        dataset_dir.joinpath(scenario_id, "info.json").write_text(dataset_info.to_json(ensure_ascii=False, indent=2))

        jumanpp = Jumanpp()
        kwja = KWJA(options=["--tasks", "word", "--model-size", "large", "--input-format", "jumanpp"])
        morphemes = []
        cursor = 0
        for phrase in flickr_sentence.phrases:
            chunk = flickr_sentence.text[cursor : phrase.span[0]]
            morphemes += jumanpp.apply_to_sentence(chunk).morphemes
            morphemes += jumanpp.apply_to_sentence(phrase.text).morphemes
            cursor = phrase.span[1]
        if flickr_sentence.text[cursor:]:
            morphemes += jumanpp.apply_to_sentence(flickr_sentence.text[cursor:]).morphemes
        knp_sentence = KNPSentence()
        knp_sentence.morphemes = morphemes
        knp_sentence.sent_id = f"{scenario_id}-00"
        knp_sentence.doc_id = scenario_id
        knp_sentence = kwja.apply_to_document(Document.from_sentences([knp_sentence]))
        knp_dir.joinpath(f"{scenario_id}.knp").write_text(knp_sentence.to_knp())

        instance_id_to_class_name: dict[str, str] = {}
        for phrase in flickr_sentence.phrases:
            instance_id_to_class_name[str(phrase.phrase_id)] = phrase.phrase_type

        image_text_annotation = ImageTextAnnotation(
            scenario_id=scenario_id,
            utterances=[
                UtteranceAnnotation(
                    text=flickr_sentence.text,
                    phrases=[],  # TODO
                )
            ],
            images=[
                ImageAnnotation(
                    image_id=flickr_image_id,
                    bounding_boxes=[
                        BoundingBox(
                            image_id=flickr_image_id,
                            instance_id=obj.name,
                            rect=Rectangle(
                                x1=obj.bndbox.xmin,
                                y1=obj.bndbox.ymin,
                                x2=obj.bndbox.xmax,
                                y2=obj.bndbox.ymax,
                            ),
                            class_name=instance_id_to_class_name[obj.name],
                        )
                        for obj in flickr_annotation.objects
                    ],
                )
            ],
        )
        annotation_dir.joinpath(f"{scenario_id}.json").write_text(
            image_text_annotation.to_json(ensure_ascii=False, indent=2)
        )
        shutil.copy(flickr_image_dir / flickr_annotation.filename, image_dir / f"{flickr_image_id}.jpg")


if __name__ == "__main__":
    main()
