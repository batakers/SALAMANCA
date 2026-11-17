"""Label-free checks used to compare model variants. Reads NO ground truth.

The competition forbids tuning on the public 5 minutes, so variants are compared on proxies built
only from the provided automatic data (Smart Tagging is algorithmic, not human-labelled) and from
our own predictions:

  team_agree   among frames inside Smart Tagging phases that name a possessing team (attacking
               phases name it directly, defending phases name the other side), the share our H/A
               label agrees with it. Computed with the model's Smart Tagging team prior switched
               OFF, so the check is not circular.
  phase_agree  the same, but one vote per tagged phase (majority of our H/A frames inside it),
               so long phases do not dominate.
  restarts     share of Smart Tagging SET PIECES windows (excluding the half-opening kick-offs the
               model uses) that contain a predicted dead-to-alive transition. The model does not
               read these windows, so this is an independent check on stoppage detection.
  poss / dur / D%   possession count, mean duration and dead-ball share on graded frames, to be
               compared with football norms, not with any ground truth.

    python tools/proxy_eval.py
"""

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession.data import load_atd, load_smart_tagging, graded_blocks  # noqa: E402
from possession.features import compute  # noqa: E402
from possession import model  # noqa: E402
from possession.collective import team_shape_features  # noqa: E402
from make_submission import segments, build_labels  # noqa: E402


def proxies(labels, tags, blocks):
    n = len(labels)
    graded = np.zeros(n, bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True

    want = np.full(n, -1)
    votes = []
    for t in tags:
        side = model.TEAM_SIDE.get(t["team"].lower())
        if side is None:
            continue
        if t["code"] in model.ATTACKING_CODES:
            team = side
        elif t["code"] in model.DEFENDING_CODES:
            team = 1 - side
        else:
            continue
        a, b = max(0, t["start"]), min(n, t["end"])
        want[a:b] = team
        seg = labels[a:b]
        h, aw = np.sum(seg == "H"), np.sum(seg == "A")
        if h + aw:
            votes.append((0 if h >= aw else 1) == team)
    m = graded & (want >= 0) & np.isin(labels, ["H", "A"])
    team_agree = float(np.mean((labels[m] == "A").astype(int) == want[m]))

    restarts, total = 0, 0
    for t in tags:
        if t["code"] != "SET PIECES" or t["type"] == "kick off (start)":
            continue
        a, b = max(0, t["start"] - 50), min(n, t["end"] + 50)
        seg = labels[a:b]
        total += 1
        restarts += bool(np.any((seg[:-1] == "D") & np.isin(seg[1:], ["H", "A"])))

    g = labels[graded]
    poss, dur = segments(labels)
    return {"team_agree": team_agree, "phase_agree": float(np.mean(votes)),
            "restarts": restarts / max(total, 1), "poss": poss, "dur": dur,
            "D%": 100 * float(np.mean(g == "D"))}


def run(variants, feat=None, tags=None, full=True):
    """full=True runs the complete two-pass pipeline (team shape included), as make_submission."""
    atd = load_atd()
    feat = feat if feat is not None else compute(atd)
    tags = tags if tags is not None else load_smart_tagging()
    blocks = graded_blocks()
    n = len(feat["bx"])
    shape = team_shape_features(atd, blocks) if full else None
    print("{:<28} {:>10} {:>11} {:>8} {:>5} {:>5} {:>5}".format(
        "variant", "team_agree", "phase_agree", "restarts", "poss", "dur", "D%"))
    out = {}
    for name, overrides in variants:
        p = dict(model.PARAMS, **overrides)
        p["tag_team_weight"] = 0.0  # never let the checked signal into the model
        labels = (build_labels(atd, feat, tags, blocks, n, p, shape) if full
                  else model.predict(feat, tags, blocks, n, p))
        r = proxies(labels, tags, blocks)
        out[name] = r
        print("{:<28} {:>10.3f} {:>11.3f} {:>8.2f} {:>5d} {:>5.1f} {:>5.1f}".format(
            name, r["team_agree"], r["phase_agree"], r["restarts"], r["poss"], r["dur"], r["D%"]))
    return out


if __name__ == "__main__":
    run([("current PARAMS", {})])
