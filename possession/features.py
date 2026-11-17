"""Per-frame features computed from the tracking. Everything in metres and m/s."""

import numpy as np

from .data import HOME, AWAY


def moving_mean(x, half_window):
    """Centred mean over +/- half_window frames, ignoring NaN. NaN where the window is empty."""
    if half_window <= 0:
        return x.astype(np.float64)
    valid = np.isfinite(x)
    vals = np.where(valid, x, 0.0).astype(np.float64)
    cs = np.concatenate(([0.0], np.cumsum(vals)))
    cn = np.concatenate(([0], np.cumsum(valid)))
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - half_window)
    hi = np.minimum(len(x), idx + half_window + 1)
    n = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)


def compute(atd):
    length, width = atd["pitch"]
    players, ball, team = atd["players"], atd["ball"], atd["team"]

    bx = ball[:, 0].astype(np.float64) * length
    by = ball[:, 1].astype(np.float64) * width
    bs = ball[:, 2].astype(np.float64)
    ball_seen = np.isfinite(bx) & np.isfinite(by)

    px = players[:, :, 0].astype(np.float64) * length
    py = players[:, :, 1].astype(np.float64) * width
    ps = players[:, :, 2].astype(np.float64)

    dist = np.hypot(px - bx[:, None], py - by[:, None])  # NaN where player or ball missing
    with np.errstate(invalid="ignore"):
        d_home = np.nanmin(np.where(team[None, :] == HOME, dist, np.nan), axis=1)
        d_away = np.nanmin(np.where(team[None, :] == AWAY, dist, np.nan), axis=1)

    # Ball displacement speed from positions (the provided speed channel is kept as well).
    vx = np.gradient(np.where(ball_seen, bx, np.nan))
    vy = np.gradient(np.where(ball_seen, by, np.nan))
    ball_speed = np.hypot(vx, vy) * 25.0
    ball_speed = np.where(np.isfinite(ball_speed), ball_speed, bs)

    # How far outside the pitch the ball is (0 inside), metres.
    out_x = np.maximum(0.0, np.maximum(-bx, bx - length))
    out_y = np.maximum(0.0, np.maximum(-by, by - width))
    ball_out = np.where(ball_seen, np.maximum(out_x, out_y), np.nan)

    # Signed distance to the nearest line (negative = inside), carried forward while the ball is
    # lost: a ball last seen on or over a line that then disappears has usually gone out.
    inside_margin = np.minimum.reduce([bx, length - bx, by, width - by])
    edge = np.where(ball_seen, -inside_margin, np.nan)
    idx = np.where(ball_seen, np.arange(len(bx)), 0)
    np.maximum.accumulate(idx, out=idx)
    last_edge = edge[idx]
    gap = np.arange(len(bx)) - idx  # frames since the ball was last seen
    edge_carried = np.where(ball_seen, edge, np.where(gap <= 25 * 20, last_edge, np.nan))

    # Length (frames) of the ball-missing run each frame belongs to; 0 where the ball is seen.
    s = ball_seen.astype(np.int8)
    d = np.diff(np.concatenate(([1], s, [1])))
    starts, ends = np.flatnonzero(d == -1), np.flatnonzero(d == 1)
    gap_len = np.zeros(len(bx), dtype=np.int32)
    for a, b in zip(starts, ends):
        gap_len[a:b] = b - a

    n_players = np.sum(np.isfinite(px), axis=1)
    with np.errstate(invalid="ignore"):
        team_speed = np.nanmean(ps, axis=1)

    return {
        "ball_seen": ball_seen,
        "bx": bx, "by": by,
        "ball_speed": ball_speed,
        "ball_out": ball_out,
        "edge_carried": edge_carried,
        "ball_gap": gap,
        "gap_len": gap_len,
        "d_home": d_home, "d_away": d_away,
        "n_players": n_players,
        "player_speed": team_speed,
    }
