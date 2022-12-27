from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(frozen=True)
class Rectangle:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int]:
        return self.x1 + self.w // 2, self.y1 + self.h // 2

    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int) -> 'Rectangle':
        return cls(x1, y1, x2, y2)

    @classmethod
    def from_cxcywh(cls, x: int, y: int, w: int, h: int) -> 'Rectangle':
        return cls.from_xyxy(x - w // 2, y - h // 2, x + w // 2, y + h // 2)

    @classmethod
    def from_xywh(cls, x: int, y: int, w: int, h: int) -> 'Rectangle':
        return cls.from_xyxy(x, y, x + w, y + h)
