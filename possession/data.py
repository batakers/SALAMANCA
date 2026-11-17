"""Load the tactical ATD and Smart Tagging into numpy, indexed by VIDEO frame (0 .. N-1).

The raw ATD is parsed once and cached as data/cache/atd_tactical.npz (data/ is gitignored), so
later runs start in about a second. Delete the cache to force a re-parse; the result is identical.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, DATA_DIR, require, source_frames, half_video_blocks  # noqa: E402

HOME, AWAY = 0, 1


def _team_of_each_track(meta_path):
    """0 for a home slot, 1 for an away slot. First <Team> in the EPTS metadata is home."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter() if t.tag.split("}")[-1] == "Team"]
    home_id = team_ids[0]
    slots = [p.get("teamId") for p in root.iter() if p.tag.split("}")[-1] == "Player"]
    return np.array([HOME if t == home_id else AWAY for t in slots], dtype=np.int8)


def _parse_raw(raw_path, n_frames, n_slots):
    """players (N, slots, 3) and ball (N, 3), columns x, y (normalised 0-1) and speed (m/s)."""
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # ATD frame_id counts from 1, video frames from 0
            if not 0 <= frame < n_frames:
                continue
            values = parts[1].replace(";", ",").split(",")
            players[frame] = np.array(values, dtype=np.float32).reshape(n_slots, 3)
            ball[frame] = np.array(parts[2].split(","), dtype=np.float32)
    return players, ball


def load_atd(game=GAME, data_dir=None, use_cache=True):
    """dict with players (N,S,3), ball (N,3), team (S,), pitch (length, width) in metres."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    cache_dir = os.path.join(data_dir or DATA_DIR, "cache")
    cache = os.path.join(cache_dir, "atd_tactical.npz")
    team = _team_of_each_track(meta_path)
    if use_cache and os.path.isfile(cache):
        z = np.load(cache)
        players, ball = z["players"], z["ball"]
    else:
        players, ball = _parse_raw(raw_path, source_frames("tactical", game), len(team))
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache, players=players, ball=ball)
    return {"players": players, "ball": ball, "team": team, "pitch": GAMES[game]["pitch"]}


def load_smart_tagging(game=GAME, data_dir=None):
    """[{code, start, end, team, type, half}] with start/end as video frames (end exclusive)."""
    path = require(["smart_tagging"], game, data_dir)[0]
    fps = GAMES[game]["fps"]
    out = []
    for inst in ET.parse(path).iter("instance"):
        tags = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        out.append({
            "code": (inst.findtext("code") or "").strip().upper(),
            "start": int(round(float(inst.findtext("start")) * fps)),
            "end": int(round(float(inst.findtext("end")) * fps)),
            "team": (tags.get("Team") or "").strip(),
            "type": (tags.get("Type") or "").strip().lower(),
            "half": (tags.get("Half") or "").strip(),
        })
    return out


def graded_blocks(game=GAME):
    """[(half, first_video_frame, last_video_frame)] -- published half boundaries."""
    return half_video_blocks(game)
