from dataclasses import dataclass

from utils.util import CamelCaseDataClassJsonMixin, Rectangle


@dataclass(frozen=True, eq=True)
class BoundingBox(CamelCaseDataClassJsonMixin):
    rect: Rectangle
    class_name: str
    confidence: float
    word_probs: list[float]

    def __hash__(self):
        return hash((self.rect, self.class_name, self.confidence, tuple(self.word_probs)))


@dataclass(frozen=True)
class MDETRPrediction(CamelCaseDataClassJsonMixin):
    bounding_boxes: list[BoundingBox]
    words: list[str]
