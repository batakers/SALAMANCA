# SALAMANCA — Ball possession by HMM segmentation

Team **SALAMANCA**'s entry to *More than a Hack 2027, Phase 1* (Metrica Sports × FC Barcelona × MWCapital): for every
frame of FC Barcelona vs Athletic Club, label who has the ball — `H`, `A`, `D` (dead ball) or `X`.

The method is a three-state segmentation (`H` / `A` / `D`) solved exactly with Viterbi: per-frame
evidence from the automatic tracking and Smart Tagging, and a fixed cost for every change of state,
so that a touch or deflection cannot create a possession and per-frame flicker is impossible. A
second pass adds team evidence from the shape of the two teams, learned by self-training (no human
labels), which is what tells the teams apart while the ball is not tracked.
Details in **[TECHNICAL.md](TECHNICAL.md)**.

## Reproduce the submission

Requires Python 3.11+ (developed on 3.14.3, Windows 11), CPU only, under two minutes.

```bash
pip install -r requirements.txt
```

Put the match data (not included — it is licensed to enrolled participants only) here:

```
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_smart_tagging.xml
```

or set `MTAH_DATA_DIR` to the folder that contains `tactical/`. The code does not read the video,
the homography or the broadcast recording. Then:

```bash
python make_submission.py --team SALAMANCA
python evaluation/validate_submission.py evaluation/results/SALAMANCA.csv
```

This writes `evaluation/results/SALAMANCA.csv` (199,319 rows), SHA-256
`3c5d190d093f942fec3bbdd186394158bf550831130e61b7fe2edffab71f7d86`. The run is deterministic: deleting
the parse cache (`data/cache/`) and re-running produces a byte-identical file.

## Layout

```
make_submission.py        the one command that produces the submission
possession/
  data.py                 ATD + Smart Tagging loaders, indexed by video frame (cached to data/cache)
  features.py             per-frame features: team-to-ball distances, ball speed, out-of-play, activity
  model.py                emission scores, switching costs, Viterbi; all parameters in PARAMS
  collective.py           team-shape classifier (self-trained, cross-fitted by half)
tools/
  proxy_eval.py           label-free checks used to compare variants (reads no ground truth)
  matched_eval.py         the same, at a matched possession count
  transitions_for_review.py  random predicted transitions to spot-check in the video (debugging
                          only; the review sheet is gitignored and never read by the pipeline)
TECHNICAL.md              the technical document
evaluation/, starter/,    the organisers' scorer, validator, baselines and loaders, unchanged
games.py, *.md            (organisers' original README is README_ORGANISERS.md)
```

## Data and licence

The match data is not in this repository and must not be committed (`data/` is gitignored). Our code
is MIT-licensed like the organisers' starter repository it builds on — see [LICENSE](LICENSE).
