"""Regenerate the submission from the provided data.

    python make_submission.py --team <team_name>

Writes evaluation/results/<team_name>.csv (199,319 rows, frame,label). Deterministic: the same
data and code always produce the same file.
"""

import argparse
import os
import sys

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN frames are expected

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from games import source_frames  # noqa: E402
from possession.data import load_atd, load_smart_tagging, graded_blocks  # noqa: E402
from possession.features import compute  # noqa: E402
from possession.model import predict, PARAMS  # noqa: E402
from possession.collective import team_shape_features, cross_fit_probability  # noqa: E402


def segments(labels):
    """(count, mean seconds) of maximal H/A runs, ignoring X."""
    lab = labels[labels != "X"]
    change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(lab)]))
    kinds = lab[starts]
    dur = (ends - starts)[np.isin(kinds, ["H", "A"])] / 25.0
    return len(dur), float(dur.mean()) if len(dur) else 0.0


def build_labels(atd, feat, tags, blocks, n, params=PARAMS, shape=None):
    """The full two-pass pipeline. `shape` (team-shape features) can be passed in to reuse it."""
    # Pass 1: ball-based model (no Smart Tagging team prior) -> pseudo-labels for self-training.
    first_pass = predict(feat, tags, blocks, n, dict(params, tag_team_weight=0.0))
    # Team-shape classifier, cross-fitted by half, trained on pass-1 labels where the ball is seen.
    if shape is None:
        shape = team_shape_features(atd, blocks)
    team_prob = cross_fit_probability(shape, first_pass, feat["ball_seen"], blocks)
    # Pass 2: the full model with team-shape evidence.
    return predict(feat, tags, blocks, n, params, team_prob)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team", default="submission", help="file name stem (your team name)")
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results"))
    args = ap.parse_args()

    n = source_frames("tactical")
    atd = load_atd()
    labels = build_labels(atd, compute(atd), load_smart_tagging(), graded_blocks(), n)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, args.team + ".csv")
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))

    graded = labels != "X"
    share = {k: round(float(np.mean(labels[graded] == k)), 3) for k in "HAD"}
    n_poss, mean_dur = segments(labels)
    print("wrote", os.path.relpath(path, ROOT))
    print("graded-frame share", share, "| possessions", n_poss, "| mean duration %.1f s" % mean_dur)


if __name__ == "__main__":
    main()
