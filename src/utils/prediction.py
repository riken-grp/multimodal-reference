from dataclasses import dataclass

from utils.util import CamelCaseDataClassJsonMixin, ImageInfo, Rectangle


@dataclass(eq=True)
class BoundingBox(CamelCaseDataClassJsonMixin):
    image_id: str
    rect: Rectangle
    confidence: float

    def __hash__(self) -> int:
        return hash((self.image_id, self.rect))


@dataclass(frozen=True, eq=True)
class RelationPrediction(CamelCaseDataClassJsonMixin):
    type: str  # ガ, ヲ, ニ, ノ, =, etc...
    image_id: str
    bounding_box: BoundingBox

    @property
    def image_idx(self) -> int:
        return int(self.image_id) - 1  # zero origin


@dataclass
class PhrasePrediction(CamelCaseDataClassJsonMixin):
    sid: str
    index: int
    text: str
    relations: list[RelationPrediction]


@dataclass
class UtterancePrediction(CamelCaseDataClassJsonMixin):
    text: str
    sids: list[str]
    phrases: list[PhrasePrediction]


@dataclass
class PhraseGroundingPrediction(CamelCaseDataClassJsonMixin):
    scenario_id: str
    images: list[ImageInfo]
    utterances: list[UtterancePrediction]
