"""Compare variants at a matched possession count (switch_team re-solved per variant).

The team-agreement proxy favours fewer switches (Smart Tagging phases are long), so variants are
compared only after adjusting the H<->A switching cost to give the same number of possessions.
Reads no ground truth.
"""
import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession.data import load_atd, load_smart_tagging, graded_blocks  # noqa: E402
from possession.features import compute  # noqa: E402
from possession import model  # noqa: E402
from tools.proxy_eval import proxies  # noqa: E402


def matched(variants, target=280, feat=None, tags=None, extra_scores=None):
    feat = feat if feat is not None else compute(load_atd())
    tags = tags if tags is not None else load_smart_tagging()
    blocks = graded_blocks(); n = len(feat["bx"])
    print("{:<30} {:>6} {:>10} {:>11} {:>8} {:>5} {:>5} {:>5}".format(
        "variant", "swT", "team_agree", "phase_agree", "restarts", "poss", "dur", "D%"))
    res = {}
    for name, ov in variants:
        lo, hi = 0.5, 40.0
        for _ in range(12):
            mid = (lo + hi) / 2
            p = dict(model.PARAMS, **ov, switch_team=mid, tag_team_weight=0.0)
            r = proxies(model.predict(feat, tags, blocks, n, p), tags, blocks)
            if r["poss"] > target: lo = mid
            else: hi = mid
        res[name] = r
        print("{:<30} {:>6.2f} {:>10.3f} {:>11.3f} {:>8.2f} {:>5d} {:>5.1f} {:>5.1f}".format(
            name, mid, r["team_agree"], r["phase_agree"], r["restarts"], r["poss"], r["dur"], r["D%"]))
    return res


if __name__ == "__main__":
    matched([("current", {}), ("radius 1.0", {"control_radius": 1.0}),
             ("radius 2.0", {"control_radius": 2.0}), ("fast_ball 5", {"fast_ball": 5.0}),
             ("fast_ball 12", {"fast_ball": 12.0})])
