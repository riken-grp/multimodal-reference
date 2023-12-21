from tasks.cohesion import CohesionAnalysis
from tasks.detic import DeticPhraseGrounding
from tasks.glip import GLIPPhraseGrounding
from tasks.mdetr import MDETRPhraseGrounding
from tasks.mmref import MultimodalReference
from tasks.mot import MultipleObjectTracking

__all__ = [
    "CohesionAnalysis",
    "MDETRPhraseGrounding",
    "GLIPPhraseGrounding",
    "DeticPhraseGrounding",
    "MultipleObjectTracking",
    "MultimodalReference",
]
