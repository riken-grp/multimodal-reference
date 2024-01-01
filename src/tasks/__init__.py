from tasks.cohesion import CohesionAnalysis
from tasks.detection.detic import DeticObjectDetection
from tasks.detection.glip import GLIPObjectDetection
from tasks.detic import DeticPhraseGrounding
from tasks.glip import GLIPPhraseGrounding
from tasks.mdetr import MDETRPhraseGrounding

# from tasks.mmref import MultimodalReference
from tasks.mot import MultipleObjectTracking

__all__ = [
    "CohesionAnalysis",
    "DeticObjectDetection",
    "GLIPObjectDetection",
    "MDETRPhraseGrounding",
    "GLIPPhraseGrounding",
    "DeticPhraseGrounding",
    "MultipleObjectTracking",
    # "MultimodalReference",
]
