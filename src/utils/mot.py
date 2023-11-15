from dataclasses import dataclass

from utils.util import CamelCaseDataClassJsonMixin, Rectangle


@dataclass(frozen=True)
class BoundingBox(CamelCaseDataClassJsonMixin):
    rect: Rectangle
    confidence: float
    class_name: str
    instance_id: int


@dataclass(frozen=True)
class Frame(CamelCaseDataClassJsonMixin):
    index: int
    bounding_boxes: list[BoundingBox]


@dataclass(frozen=True)
class DetectionLabels(CamelCaseDataClassJsonMixin):
    frames: list[Frame]
    class_names: list[str]
