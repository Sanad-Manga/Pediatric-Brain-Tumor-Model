"""Plain-text clinical summary for a segmented case.

Every field is required from the caller. The previous version supplied defaults
(48.2 cm3 volume, 24.2% ET, 96.4% confidence) that silently produced a
confident-looking report for a case nobody had measured, and stamped it
"Verified & Validated". A missing measurement now renders as "not measured"
rather than as a plausible number.
"""

from __future__ import annotations


def _fmt(value, suffix: str = "", digits: int = 1) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def generate_report_text(patient_id: str, statistics: dict) -> str:
    """Render the summary. `statistics` keys are all optional but never faked."""
    s = statistics or {}
    lines = [
        "NEUROFED AI - SEGMENTATION SUMMARY",
        "-------------------------------------------",
        f"Subject ID: {patient_id}",
        "Dataset: BraTS-PEDs pediatric cohort",
        "",
        f"Slice / plane:        {_fmt(s.get('slice_label'))}",
        f"Predicted tumour area:{_fmt(s.get('tumor_pixels'), ' px', 0):>12}",
        f"Enhancing (ET):       {_fmt(s.get('et'), '%')}",
        f"Non-enhancing core:   {_fmt(s.get('netc'), '%')}",
        f"Cystic component:     {_fmt(s.get('cc'), '%')}",
        f"Peritumoral edema:    {_fmt(s.get('ed'), '%')}",
        "",
        f"Dice vs ground truth: ET {_fmt(s.get('dice_et'), '', 3)} | "
        f"TC {_fmt(s.get('dice_tc'), '', 3)} | WT {_fmt(s.get('dice_wt'), '', 3)}",
        f"Mean softmax confidence: {_fmt(s.get('confidence'), '%')}",
        f"Checkpoint: {_fmt(s.get('checkpoint'))} "
        f"({_fmt(s.get('epochs'), ' epochs', 0)})",
        "",
        "RESEARCH USE ONLY - NOT A DIAGNOSIS.",
        "Output of a small from-scratch U-Net; not a clinically validated system.",
    ]
    return "\n".join(lines) + "\n"


def generate_pdf_summary(patient_id: str, statistics: dict) -> bytes:
    """Bytes for download. Kept for callers that expect the old name.

    Note: this emits plain text, not PDF — it always did. The name is retained
    so existing call sites keep working; prefer `generate_report_text`.
    """
    return generate_report_text(patient_id, statistics).encode("utf-8")
