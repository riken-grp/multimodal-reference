from dataclasses import dataclass

from utils.util import CamelCaseDataClassJsonMixin, Rectangle


@dataclass(frozen=True)
class BoundingBox(CamelCaseDataClassJsonMixin):
    image_id: str
    rect: Rectangle
    confidence: float


@dataclass(frozen=True)
class GLIPPhrasePrediction(CamelCaseDataClassJsonMixin):
    phrase_index: int
    phrase: str
    bounding_boxes: list[BoundingBox]


@dataclass(frozen=True)
class GLIPPrediction(CamelCaseDataClassJsonMixin):
    doc_id: str
    phrase_predictions: list[GLIPPhrasePrediction]
    phrases: list[str]
