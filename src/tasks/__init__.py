from tasks.cohesion import CohesionAnalysis
from tasks.detection.detic import DeticObjectDetection
from tasks.detection.glip import GLIPObjectDetection
from tasks.grounding.detic import DeticPhraseGrounding
from tasks.grounding.glip import GLIPPhraseGrounding
from tasks.grounding.mdetr import MDETRPhraseGrounding
from tasks.grounding.vlm_som import SoMPhraseGrounding
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
    "SoMPhraseGrounding",
]
