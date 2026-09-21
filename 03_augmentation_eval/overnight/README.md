# Unattended-training and ensemble tooling

Scripts used for the overnight rounds on the RTX 3060 training PC.
**Paths are machine-specific** (`C:\Users\ahmed\...`, `D:\NeuroPeds AI\...`): edit the constants at the
top of each file before using them elsewhere.

| File | Purpose |
|------|---------|
| `orchestrate.py` | Helpers every round script imports: `preflight()` (refuses to run on CPU-only torch; **holds Windows awake** via `SetThreadExecutionState`), `run_with_timeout()` (wall-clock deadline + `taskkill /T` of the whole process tree), `held_out_eval()`, `restart_streamlit()`. |
| `build_ensemble_checkpoint.py` | Packs two checkpoints into one self-contained ensemble `best.pt` the app can load. |
| `score_ensemble_cli.py` | Held-out `--eval-plane both` Dice for an ensemble of N checkpoints (`run.py eval` only scores single models). |

Why the sleep guard and wall-clock timeout exist: see "Lessons from this round" in `HANDOFF.md`.
