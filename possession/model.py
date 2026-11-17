"""A three-state segmentation (H, A, D) solved exactly with Viterbi.

Every frame contributes evidence (a score) for each state; every change of state costs a fixed
penalty. Viterbi returns the label sequence with the highest total score. This encodes the two
things the challenge cares about:

  * possession changes only on CONTROLLED possession -- a touch, a deflection or a lost aerial
    duel produces a few frames of opposing evidence, which is less than the cost of switching
    twice, so it does not create a possession;
  * a possession is an interval -- per-frame flicker is impossible by construction.

Where the ball is not tracked (41% of graded frames) there is simply no team evidence, so the
current state carries on until the ball is seen again. A missing ball is never treated as a dead
ball on its own.

All parameters are set a priori from football and tracking reasoning (see TECHNICAL.md), not fitted
on any ground truth.
"""

import numpy as np

from .features import moving_mean

H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])

PARAMS = {
    # -- team control --------------------------------------------------------------------------
    "control_radius": 1.5,      # m; Gaussian scale of "this player is on the ball" (allows for
                                # ~1 m of automatic-tracking noise on ball and players)
    "fast_ball": 8.0,           # m/s; above this the ball is travelling, proximity means little
    "team_weight": 0.3,         # score per frame of soft closeness (secondary to control runs)
    "switch_team": 4.5,         # cost of H<->A; a temporary excursion must out-score 2x this
    # controlled possession (exclusive and sustained) -- the main team evidence, see emissions()
    "ctrl_weight": 1.0,         # score per frame of a sustained control run
    "ctrl_radius": 1.0,         # m; player of the team this close to the ball ...
    "ctrl_exclusive": 1.5,      # m; ... and no opponent this close ...
    "ctrl_ball_speed": 6.0,     # m/s; ... with the ball no faster than a dribble ...
    "ctrl_min_run": 5,          # frames; ... for at least 0.2 s (a touch lasts 1-3 frames)
    "ctrl_bridge": 2,           # frames; tracking jitter tolerated inside a run
    # -- dead ball -----------------------------------------------------------------------------
    "switch_dead": 16.0,        # cost of alive<->dead, paid twice per stoppage
    "out_margin": 0.3,          # m; ball this far beyond a line counts as out of play
    "out_weight": 1.0,          # D score per frame with the ball out of play
    "lost_edge": 0.5,           # m; ball lost while last seen within this of a line (or beyond)
    "lost_out_weight": 0.5,     # D score per frame while it stays lost
    "activity_ref": 1.4,        # m/s; mean player speed below this looks like a stoppage
    "activity_scale": 0.5,      # m/s; how quickly that evidence saturates
    "activity_weight": 0.35,    # max D score per frame from player activity
    "long_gap": 125,            # frames; a ball-missing run at least this long (5 s) ...
    "gap_weight": 0.3,          # ... adds this D score per frame inside it
    "still_ball": 0.7,          # m/s; a ball slower than this for a second is being placed
    "still_weight": 0.3,
    "moving_ball": 5.0,         # m/s; a ball faster than this inside the pitch is in play
    "moving_weight": 0.3,
    # -- team shape (possession/collective.py) ---------------------------------------------------
    "shape_weight_seen": 0.1,   # score per frame per unit of (P(home) - 0.5), ball tracked
    "shape_weight_missing": 0.3,  # same, ball not tracked -- the only team evidence there
    # -- Smart Tagging priors --------------------------------------------------------------------
    "tag_team_weight": 0.05,    # small per-frame nudge towards the team a phase names
    "tag_dead_weight": 2.0,     # before a half's kick-off, and between a goal and its kick-off
}

# Smart Tagging codes and which side of them is in possession.
ATTACKING_CODES = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
                   "BALL IN FINAL THIRD", "BALL IN THE BOX"}
DEFENDING_CODES = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                   "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}
TEAM_SIDE = {"barcelona": H, "atletic club": A, "athletic club": A}


def sustained(mask, min_run, bridge=0):
    """mask with only its runs of at least min_run frames kept, after closing gaps <= bridge."""
    m = mask.astype(np.int8)
    if bridge > 0:  # fill short holes (tracking jitter) between two True frames
        d = np.diff(np.concatenate(([1], m, [1])))
        starts, ends = np.flatnonzero(d == -1), np.flatnonzero(d == 1)
        for a, b in zip(starts, ends):
            if 0 < a and b < len(m) and b - a <= bridge:
                m[a:b] = 1
    d = np.diff(np.concatenate(([0], m, [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out = np.zeros(len(m), dtype=np.float64)
    for a, b in zip(starts, ends):
        if b - a >= min_run:
            out[a:b] = 1.0
    return out


def emissions(feat, tags, blocks, p=PARAMS, team_prob=None):
    """(N, 3) per-frame scores for H, A, D. team_prob: optional P(home) from team shape."""
    n = len(feat["bx"])
    s = np.zeros((n, 3))

    if team_prob is not None:
        w = np.where(feat["ball_seen"], p["shape_weight_seen"], p["shape_weight_missing"])
        q = np.nan_to_num(team_prob, nan=0.5) - 0.5
        s[:, H] += w * q
        s[:, A] -= w * q

    # Team control: Gaussian closeness of each team's nearest player, damped for a fast ball.
    seen = feat["ball_seen"]
    out = np.nan_to_num(feat["ball_out"], nan=0.0)
    in_play_area = seen & (out <= p["out_margin"])
    speed = np.nan_to_num(feat["ball_speed"], nan=0.0)
    damp = 1.0 / (1.0 + (speed / p["fast_ball"]) ** 4)
    r = p["control_radius"]
    c_home = np.exp(-0.5 * (np.nan_to_num(feat["d_home"], nan=1e3) / r) ** 2) * damp
    c_away = np.exp(-0.5 * (np.nan_to_num(feat["d_away"], nan=1e3) / r) ** 2) * damp
    s[:, H] += np.where(in_play_area, p["team_weight"] * c_home, 0.0)
    s[:, A] += np.where(in_play_area, p["team_weight"] * c_away, 0.0)

    # Controlled possession, as CHALLENGE.md defines it: a player of one team on a ball that is
    # not travelling, with no opponent as close, SUSTAINED for a minimum time. A touch, a header or
    # a deflection lasts a frame or three and earns nothing here.
    if p["ctrl_weight"] > 0:
        d_h = np.nan_to_num(feat["d_home"], nan=1e3)
        d_a = np.nan_to_num(feat["d_away"], nan=1e3)
        slow = speed < p["ctrl_ball_speed"]
        ctrl_h = in_play_area & slow & (d_h < p["ctrl_radius"]) & (d_a > p["ctrl_exclusive"])
        ctrl_a = in_play_area & slow & (d_a < p["ctrl_radius"]) & (d_h > p["ctrl_exclusive"])
        s[:, H] += p["ctrl_weight"] * sustained(ctrl_h, p["ctrl_min_run"], p["ctrl_bridge"])
        s[:, A] += p["ctrl_weight"] * sustained(ctrl_a, p["ctrl_min_run"], p["ctrl_bridge"])

    # Dead ball, expressed as a D score (positive = looks dead, negative = looks alive).
    dead = np.zeros(n)
    dead += np.where(seen & (out > p["out_margin"]), p["out_weight"], 0.0)
    # Ball lost right after being seen on/over a line (within lost_edge m of it): likely out.
    edge = feat["edge_carried"]
    lost_near_line = (~seen) & np.isfinite(edge) & (edge > -p["lost_edge"])
    dead += np.where(lost_near_line, p["lost_out_weight"], 0.0)
    # The tracker tends to drop the ball for the whole of a stoppage: a long ball-missing run is
    # evidence of a dead ball, and its edges mark when play stopped and restarted.
    long_gap = feat["gap_len"] >= p["long_gap"]
    dead += np.where(long_gap, p["gap_weight"], 0.0)
    activity = moving_mean(feat["player_speed"], 25)
    act = np.clip((p["activity_ref"] - np.nan_to_num(activity, nan=p["activity_ref"]))
                  / p["activity_scale"], -1.0, 1.0)
    dead += p["activity_weight"] * act
    ball_speed_1s = moving_mean(np.where(seen, feat["ball_speed"], np.nan), 12)
    dead += np.where(np.isfinite(ball_speed_1s) & (ball_speed_1s < p["still_ball"]),
                     p["still_weight"], 0.0)
    dead -= np.where(in_play_area & (speed > p["moving_ball"]), p["moving_weight"], 0.0)
    s[:, D] += dead

    # Smart Tagging.
    team_prior = np.zeros((n, 2))
    for t in tags:
        side = TEAM_SIDE.get(t["team"].lower())
        if side is None:
            continue
        a, b = max(0, t["start"]), min(n, t["end"])
        if t["code"] in ATTACKING_CODES:
            team_prior[a:b, side] = 1.0
        elif t["code"] in DEFENDING_CODES:
            team_prior[a:b, 1 - side] = 1.0
    s[:, H] += p["tag_team_weight"] * team_prior[:, H]
    s[:, A] += p["tag_team_weight"] * team_prior[:, A]

    # Before the opening kick-off of a half the ball is not in play. Only the "Kick Off (Start)"
    # tags mark a half's opening; after-goal kick-offs are left to the tracking evidence.
    half_starts = sorted(t["start"] for t in tags
                         if t["code"] == "SET PIECES" and t["type"] == "kick off (start)")
    for _, first, last in blocks:
        ko = [k for k in half_starts if first - 25 * 60 <= k <= last]
        if ko and ko[0] > first:
            s[first:ko[0], D] += p["tag_dead_weight"]
    return s


def viterbi(scores, p=PARAMS):
    """Best path through H/A/D with switching costs. scores: (T, 3)."""
    T = len(scores)
    cost = np.array([
        [0.0, p["switch_team"], p["switch_dead"]],
        [p["switch_team"], 0.0, p["switch_dead"]],
        [p["switch_dead"], p["switch_dead"], 0.0],
    ])
    back = np.zeros((T, 3), dtype=np.int8)
    v = scores[0].copy()
    sc = scores.tolist()
    c = cost.tolist()
    v0, v1, v2 = v.tolist()
    for t in range(1, T):
        cand = [[v0 - c[0][j], v1 - c[1][j], v2 - c[2][j]] for j in range(3)]
        row = sc[t]
        nv = []
        for j in range(3):
            cj = cand[j]
            k = 0 if cj[0] >= cj[1] and cj[0] >= cj[2] else (1 if cj[1] >= cj[2] else 2)
            back[t, j] = k
            nv.append(cj[k] + row[j])
        v0, v1, v2 = nv
    path = np.zeros(T, dtype=np.int8)
    last = [v0, v1, v2]
    path[-1] = int(np.argmax(last))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def predict(feat, tags, blocks, n_frames, p=PARAMS, team_prob=None):
    """Labels for every video frame: H/A/D inside the graded halves, X outside them."""
    scores = emissions(feat, tags, blocks, p, team_prob)
    labels = np.full(n_frames, "X", dtype="<U1")
    for _, first, last in blocks:
        path = viterbi(scores[first:last + 1], p)
        labels[first:last + 1] = LABELS[path]
    return labels
