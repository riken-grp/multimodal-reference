from dataclasses import dataclass

from utils.util import CamelCaseDataClassJsonMixin, Rectangle


@dataclass(frozen=True)
class BoundingBox(CamelCaseDataClassJsonMixin):
    image_id: str
    rect: Rectangle
    confidence: float


@dataclass(frozen=True)
class GLIPPhrasePrediction(CamelCaseDataClassJsonMixin):
    index: int
    text: str
    bounding_boxes: list[BoundingBox]


@dataclass(frozen=True)
class GLIPPrediction(CamelCaseDataClassJsonMixin):
    doc_id: str
    image_id: str
    phrases: list[GLIPPhrasePrediction]
