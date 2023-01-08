from dataclasses import dataclass

from utils.util import CamelCaseDataClassJsonMixin, Rectangle


@dataclass(frozen=True)
class BoundingBox(CamelCaseDataClassJsonMixin):
    image_id: str
    instance_id: str
    rect: Rectangle
    class_name: str


@dataclass(frozen=True)
class Phrase2ObjectRelation(CamelCaseDataClassJsonMixin):
    type: str  # ガ, ヲ, ニ, ノ, =, etc...
    instance_id: str


@dataclass(frozen=True)
class PhraseAnnotation(CamelCaseDataClassJsonMixin):
    text: str
    relations: list[Phrase2ObjectRelation]


@dataclass(frozen=True)
class ImageAnnotation(CamelCaseDataClassJsonMixin):
    image_id: str
    bounding_boxes: list[BoundingBox]


@dataclass(frozen=True)
class UtteranceAnnotation(CamelCaseDataClassJsonMixin):
    text: str
    phrases: list[PhraseAnnotation]


@dataclass(frozen=True)
class ImageTextAnnotation(CamelCaseDataClassJsonMixin):
    scenario_id: str
    images: list[ImageAnnotation]
    utterances: list[UtteranceAnnotation]
