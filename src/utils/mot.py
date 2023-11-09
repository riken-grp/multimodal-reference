from dataclasses import dataclass

from utils.util import CamelCaseDataClassJsonMixin


@dataclass(frozen=True)
class BoundingBox(CamelCaseDataClassJsonMixin):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    instance_id: int

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1


@dataclass(frozen=True)
class Frame(CamelCaseDataClassJsonMixin):
    bounding_boxes: list[BoundingBox]


@dataclass(frozen=True)
class DetectionLabels(CamelCaseDataClassJsonMixin):
    frames: list[Frame]
    classes: list[str]
