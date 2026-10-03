"""Regression check for 05_frontend_demo: every page compiles AND renders content, no exceptions; comparison page
numbers equal meta.json for every patient (R* and hi-res); flag tables equal review_spots.json.
Run from 05_frontend_demo: python check_app.py"""
import ast
import json
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

bad = 0
for page in ["Home.py", *sorted(str(p) for p in Path("pages").glob("*.py"))]:
    try:
        ast.parse(Path(page).read_text(encoding="utf-8"))
    except SyntaxError as e:
        print("SYNTAX", page, e); bad += 1; continue
    at = AppTest.from_file(page, default_timeout=120); at.run()
    n = len(at.main.children)
    print(f"{page}: exceptions {len(at.exception)}, top-level elements {n}")
    bad += bool(at.exception) or n == 0

at = AppTest.from_file("pages/Model_Comparison.py", default_timeout=120); at.run()
for sid in at.selectbox[0].options:
    at.selectbox[0].set_value(sid).run()
    meta = json.load(open(f"comparison_cache/{sid}/meta.json"))
    want = [f"{meta['regions'][k][r]['dice']:.4f}" for k in ("2d", "3d", "rstar") for r in ("ET", "NC", "WT")]
    spots = json.load(open(f"comparison_cache/{sid}/review_spots.json"))
    ok = not at.exception and [m.value for m in at.metric] == want and sum(len(d.value) for d in at.dataframe) == len(spots)
    at.toggle[-1].set_value(True).run()
    hw = [f"{meta['regions']['rstar_hires'][r]['dice']:.4f}" for r in ("ET", "NC", "WT")]
    ok &= not at.exception and [m.value for m in at.metric][6:] == hw
    at.toggle[-1].set_value(False).run()
    bad += not ok
    if not ok:
        print("BAD", sid)
print("comparison patients checked:", len(at.selectbox[0].options))
print("ALL OK" if bad == 0 else f"{bad} PROBLEMS")
sys.exit(1 if bad else 0)
