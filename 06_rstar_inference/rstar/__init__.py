"""R* inference: the 2D+3D fusion recipe for pediatric brain tumour segmentation (see SPEC.md).

Importing this package does not touch sys.path and does not import section 01 or 03 code; that happens when models are loaded.
"""
from .config import SEQUENCES, RStarConfig
from .contract import ContractError
from .guards import SelfCheckError
from .models import ModelIntegrityError
from .pipeline import RStarResult, RStarSegmenter

__all__ = ["RStarSegmenter", "RStarConfig", "RStarResult", "ContractError", "ModelIntegrityError", "SelfCheckError", "SEQUENCES"]
