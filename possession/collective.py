"""Which team has the ball, judged from the shape and movement of the 22 players alone.

Trained by self-training, with no human labels: the pseudo-labels are our own model's H/A output on
frames where the ball IS tracked; the classifier then supplies team evidence everywhere, which
matters most where the ball is not tracked. Cross-fitted by half -- the classifier applied to a half
was trained only on the other half -- so no frame is scored by a model that saw its pseudo-label.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .data import HOME, AWAY
from .features import moving_mean


def team_shape_features(atd, blocks):
    """(N, F) collective features. x is flipped in the second half so each team attacks the same
    way in both halves (teams change ends at half-time)."""
    length, width = atd["pitch"]
    players, team = atd["players"], atd["team"]
    n = len(players)
    px = players[:, :, 0].astype(np.float64) * length
    py = players[:, :, 1].astype(np.float64) * width
    ps = players[:, :, 2].astype(np.float64)
    for half, first, last in blocks:
        if half == 2:
            px[first:last + 1] = length - px[first:last + 1]

    cols = []
    for side in (HOME, AWAY):
        m = team == side
        x, y, s = px[:, m], py[:, m], ps[:, m]
        cx = np.nanmean(x, axis=1)
        cy = np.nanmean(y, axis=1)
        sx = np.nanstd(x, axis=1)
        sy = np.nanstd(y, axis=1)
        q = np.nanpercentile(x, [10, 90], axis=1)
        vx = np.gradient(moving_mean(cx, 6)) * 25.0
        vy = np.gradient(moving_mean(cy, 6)) * 25.0
        sp = np.nanmean(s, axis=1)
        cols += [cx, cy, sx, sy, q[0], q[1], moving_mean(vx, 12), moving_mean(vy, 12),
                 moving_mean(sp, 12)]
    f = np.column_stack(cols)
    k = len(cols) // 2
    return np.column_stack([f, f[:, :k] - f[:, k:]])


def cross_fit_probability(x, labels, ball_seen, blocks, seed=0):
    """P(home in possession) for every graded frame, cross-fitted by half. NaN elsewhere."""
    n = len(labels)
    prob = np.full(n, np.nan)
    halves = [(first, last) for _, first, last in blocks]
    for i, (first, last) in enumerate(halves):
        train = np.zeros(n, bool)
        for j, (a, b) in enumerate(halves):
            if j != i:
                train[a:b + 1] = True
        train &= ball_seen & np.isin(labels, ["H", "A"])
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                             l2_regularization=1.0, early_stopping=False,
                                             random_state=seed)
        clf.fit(x[train], (labels[train] == "H").astype(int))
        prob[first:last + 1] = clf.predict_proba(x[first:last + 1])[:, 1]
    return prob
