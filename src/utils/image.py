from dataclasses import dataclass

from dataclasses_json import DataClassJsonMixin


@dataclass
class BoundingBox(DataClassJsonMixin):
    image_id: str
    id: str
    bb: list[int]  # [x, y, w, h]
    class_name: str
    auto_flg: bool  # true: 自動推定, false: 手動指定


@dataclass
class Caption(DataClassJsonMixin):
    image_id: str
    id: str  # same as image_id
    text: str


@dataclass
class ImageRelation(DataClassJsonMixin):
    image_id: str
    id: str
    from_: str
    to: str
    relation: str  # class name of the relation
    auto_flg: bool  # true: 自動推定, false: 手動指定


@dataclass
class Image2TextRelation(DataClassJsonMixin):
    image_id: str
    id: str
    visual_id: str
    caption_id: str
    caption_region: list[int]


@dataclass
class ImageAnnotation(DataClassJsonMixin):
    image_id: str
    bounding_boxes: list[BoundingBox]
    caption: Caption
    relations: list[ImageRelation]
    i2t_relations: list[Image2TextRelation]
