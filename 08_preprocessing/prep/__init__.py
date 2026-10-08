"""NeuroPeds preprocessing for hospital scans. Part 1: rigid alignment of a patient's sequences to their T1c."""
from .align import SEQUENCES, align_sequences, align_to_reference, residual_error_mm

__all__ = ["SEQUENCES", "align_sequences", "align_to_reference", "residual_error_mm"]
