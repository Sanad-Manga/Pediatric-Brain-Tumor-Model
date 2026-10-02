"""Review-flag prototype (SPEC.md). Reads only the local cache written by precompute.py.

    streamlit run 07_review_prototype/app.py
    (cache path: env NEUROPEDS_PROTOTYPE_CACHE, default D:/NeuroPeds AI/review_prototype)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from scipy import ndimage

CACHE = Path(os.environ.get("NEUROPEDS_PROTOTYPE_CACHE", r"D:\NeuroPeds AI\review_prototype"))
# Label colours (RGB): 1 enhancing, 2 non-enhancing core, 3 cyst, 4 edema; outlines for review / specks / expert.
COLOURS = {1: (220, 40, 60), 2: (240, 170, 40), 3: (150, 90, 220), 4: (60, 170, 120)}
REVIEW = (0, 230, 255)
SPECK = (255, 255, 255)
EXPERT = (255, 230, 0)
ALPHA = 0.45

st.set_page_config(page_title="NeuroPeds review-flag prototype", layout="wide")
st.warning("Research prototype - not for clinical use. Validated on US BraTS-PEDs data only.")
st.title("Enhancing-tumour review flags: prototype")

if not (CACHE / "patients.json").is_file():
    st.error(f"No prototype cache at `{CACHE}`. Run: `python 07_review_prototype/precompute.py --out \"{CACHE}\"` "
             "(or set NEUROPEDS_PROTOTYPE_CACHE).")
    st.stop()


@st.cache_data
def load(sid: str):
    d = CACHE / sid
    arr = lambda name, key: np.load(d / f"{name}.npz")[key]
    js = lambda name: json.loads((d / f"{name}.json").read_text(encoding="utf-8"))
    return dict(t1c=arr("t1c", "img"), t2f=arr("t2f", "img"), gt=arr("gt", "labels"), today=arr("today", "labels"),
                flagged=arr("flagged", "labels"), review=arr("review_mask", "mask"), specks=arr("specks", "mask"),
                review_spots=js("review_spots"), speck_list=js("specks"), meta=js("meta"))


def outline(mask2d: np.ndarray) -> np.ndarray:
    return mask2d & ~ndimage.binary_erosion(mask2d, iterations=1)


def render(bg: np.ndarray, labels: np.ndarray, outlines: list[tuple[np.ndarray, tuple]]) -> Image.Image:
    rgb = np.repeat(bg[..., None], 3, axis=2).astype(np.float32)
    for lab, col in COLOURS.items():
        m = labels == lab
        rgb[m] = (1 - ALPHA) * rgb[m] + ALPHA * np.array(col, np.float32)
    for m, col in outlines:
        rgb[outline(m)] = col
    # BraTS arrays are LPS: axis 0 runs to the patient's left, axis 1 to posterior. Rows = axis 1 puts anterior
    # at the top and the patient's left on the image right (radiological convention).
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8).transpose(1, 0, 2)).resize((480, 480), Image.NEAREST)


patients = json.loads((CACHE / "patients.json").read_text(encoding="utf-8"))
with st.sidebar:
    sid = st.selectbox("Patient", patients, key="patient")
    seq = st.radio("Background", ["T1c (contrast)", "FLAIR"], key="seq")
    show_expert = st.checkbox("Show the expert's enhancing tumour (yellow outline)", key="expert")
    show_specks = st.checkbox("Show specks < 50 voxels (white)", key="specks")
    reveal = st.checkbox("Reveal whether each flagged spot is real (uses the expert labels)", key="reveal")
    st.caption("Review rule: an enhancing spot of at least 50 voxels is flagged when the model's average confidence "
               "that it is enhancing is below 70%. Today's app instead removes ALL enhancing tumour when a patient's "
               "total is under 500 mm³.")

d = load(sid)
meta = d["meta"]
z_with_et = np.where((d["flagged"] == 1).any(axis=(0, 1)) | (d["gt"] > 0).any(axis=(0, 1)))[0]
default_z = int(np.median(z_with_et)) if len(z_with_et) else 77
if st.session_state.get("slider_for") != sid:          # new patient: centre the slider on the tumour
    st.session_state["z"] = default_z
    st.session_state["slider_for"] = sid
z = st.slider("Axial slice", 0, d["t1c"].shape[2] - 1, key="z")

c1, c2, c3 = st.columns(3)
c1.metric("Spots kept as enhancing", meta["n_kept"])
c2.metric("Spots flagged for review", meta["n_review"])
c3.metric("Specks < 50 voxels", meta["n_specks"])

bg = (d["t1c"] if seq.startswith("T1c") else d["t2f"])[:, :, z]
expert = [(d["gt"][:, :, z] == 1, EXPERT)] if show_expert else []
left, right = st.columns(2)
with left:
    st.subheader("Today (500 mm³ rule)")
    st.image(render(bg, d["today"][:, :, z], expert), use_container_width=True)
with right:
    st.subheader("With review flags")
    outs = expert + [(d["review"][:, :, z] > 0, REVIEW)]
    if show_specks:
        outs.append((d["specks"][:, :, z] > 0, SPECK))
    flagged = d["flagged"][:, :, z].copy()
    if not show_specks:
        flagged[(d["specks"][:, :, z] > 0)] = 0      # specks hidden: not drawn as enhancing either
    st.image(render(bg, flagged, outs), use_container_width=True)
st.caption("Colours: red = enhancing, orange = non-enhancing core, purple = cyst, green = edema. "
           "Cyan outline = flagged for review; white = speck; yellow = expert's enhancing tumour.")

st.subheader("Spots flagged for review")
rows = d["review_spots"] + (d["speck_list"] if show_specks else [])
if not rows:
    st.info("No spots flagged for review for this patient.")
for e in rows:
    kind = "speck" if e["reason"] == "speck" else "review"
    lo, hi = e["slices"]
    cols = st.columns([3, 2, 2, 2, 2, 2])
    cols[0].markdown(f"**{kind} spot {e['spot_id']}** - slices {lo}-{hi}")
    cols[1].write(f"{e['voxels']} voxels")
    cols[2].write(f"confidence {e['mean_et_prob']:.0%}")
    cols[3].write("agreement " + ("-" if e["models_agree"] is None else f"{e['models_agree']:.0%}"))
    cols[4].write(("**real**" if e["real"] else "**false**") if reveal else "")
    # on_click runs BEFORE the next script run, the only point where a drawn slider's value may be changed
    cols[5].button("Go to spot", key=f"go-{kind}-{e['spot_id']}", on_click=st.session_state.__setitem__,
                   args=("z", (lo + hi) // 2))
