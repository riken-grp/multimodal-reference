from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, Union

from dataclasses_json import DataClassJsonMixin, LetterCase, config

number = Union[int, float]


class CamelCaseDataClassJsonMixin(DataClassJsonMixin):
    dataclass_json_config = config(letter_case=LetterCase.CAMEL)["dataclasses_json"]  # type: ignore


@dataclass
class ImageInfo(CamelCaseDataClassJsonMixin):
    id: str
    path: str
    time: int


@dataclass
class UtteranceInfo(CamelCaseDataClassJsonMixin):
    text: str
    sids: list[str]
    start: int
    end: int
    duration: int
    speaker: str
    image_ids: list[str]


@dataclass
class DatasetInfo(CamelCaseDataClassJsonMixin):
    scenario_id: str
    utterances: list[UtteranceInfo]
    images: list[ImageInfo]


@dataclass(frozen=True, eq=True)
class Rectangle(CamelCaseDataClassJsonMixin):
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self) -> int:
        return abs(self.x2 - self.x1)

    @property
    def h(self) -> int:
        return abs(self.y2 - self.y1)

    @property
    def center(self) -> tuple[int, int]:
        return self.x1 + self.w // 2, self.y1 + self.h // 2

    @property
    def area(self) -> int:
        return self.w * self.h

    @classmethod
    def from_xyxy(cls, x1: number, y1: number, x2: number, y2: number) -> "Rectangle":
        return cls(*map(int, (x1, y1, x2, y2)))

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return min(self.x1, self.x2), min(self.y1, self.y2), max(self.x1, self.x2), max(self.y1, self.y2)

    @classmethod
    def from_cxcywh(cls, x: number, y: number, w: number, h: number) -> "Rectangle":
        if w < 0 or h < 0:
            raise ValueError("w and h must be positive")
        return cls.from_xyxy(x - w / 2, y - h / 2, x + w / 2, y + h / 2)

    def to_cxcywh(self) -> tuple[int, int, int, int]:
        return self.center[0], self.center[1], self.w, self.h

    @classmethod
    def from_xywh(cls, top_left_x: number, top_left_y: number, w: number, h: number) -> "Rectangle":
        if w < 0 or h < 0:
            raise ValueError("w and h must be positive")
        return cls.from_xyxy(top_left_x, top_left_y, top_left_x + w, top_left_y + h)

    def to_xywh(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.w, self.h

    def __and__(self, other: Any) -> "Rectangle":
        if isinstance(other, type(self)) is False:
            raise TypeError(f"unsupported operand type(s) for &: '{type(self)}' and '{type(other)}'")
        xyxy1, xyxy2 = self.to_xyxy(), other.to_xyxy()
        xyxy = max(xyxy1[0], xyxy2[0]), max(xyxy1[1], xyxy2[1]), min(xyxy1[2], xyxy2[2]), min(xyxy1[3], xyxy2[3])
        return Rectangle.from_xyxy(xyxy[0], xyxy[1], max(xyxy[0], xyxy[2]), max(xyxy[1], xyxy[3]))


def box_iou(box1: Rectangle, box2: Rectangle) -> float:
    if box1.area == 0 or box2.area == 0:
        return 0
    intersect: int = (box1 & box2).area
    return intersect / (box1.area + box2.area - intersect)


class IdMapper:
    """Consistently map ids of any type to integers."""

    def __init__(self):
        self._id_to_int: dict[Hashable, int] = {}
        self._int_to_id: dict[int, Hashable] = {}
        self._next_int: int = 0

    def __len__(self) -> int:
        return len(self._id_to_int)

    def __contains__(self, id_: object) -> bool:
        return id_ in self._id_to_int

    def __getitem__(self, id_: object) -> int:
        if id_ not in self._id_to_int:
            self._id_to_int[id_] = self._next_int
            self._int_to_id[self._next_int] = id_
            self._next_int += 1
        return self._id_to_int[id_]

    def map(self, id_: object) -> int:
        return self[id_]
