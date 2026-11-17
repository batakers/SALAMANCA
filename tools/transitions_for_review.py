"""List a random sample of predicted transitions to spot-check against the tactical video.

    python tools/transitions_for_review.py                       # model_v4.csv, 20 moments
    python tools/transitions_for_review.py --file evaluation/results/<team>.csv --n 30 --seed 1

Writes review/transitions_review.csv (open it in Excel / Google Sheets) with, for each moment:
the video time to seek to in BAR-ATH_tactical.mp4, the approximate match clock, what the model
says happens, and empty `verdict` / `note` columns to fill in while watching.

WHAT THIS IS FOR, AND WHAT IT IS NOT (RULES.md): watching the video to understand where the model
goes wrong and to design better *general* rules is allowed. Turning what you see into labels or
corrections for specific frames, or into constants tied to specific moments of this game, is not.
The review file is therefore gitignored and must never be read by the pipeline.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402

FPS = 25
KICKOFF_2ND = 113742  # second-half kick-off video frame, published in FAQ.md / DATA.md
NAMES = {"H": "Barcelona", "A": "Athletic", "D": "bola mati"}
KINDS = {
    "team": "pergantian penguasaan (H<->A)",
    "dead": "bola menjadi mati (hidup->D)",
    "restart": "restart (D->hidup)",
}


def video_time(frame):
    s = frame / FPS
    return "{:d}:{:02d}:{:05.2f}".format(int(s // 3600), int(s % 3600 // 60), s % 60)


def match_clock(frame, blocks):
    (_, first1, last1), (_, first2, last2) = blocks
    if first1 <= frame <= last1:
        s, base = (frame - first1) / FPS, 0
    elif first2 <= frame <= last2:
        s, base = (frame - KICKOFF_2ND) / FPS, 45 * 60
        if s < 0:
            return "sebelum kick-off babak 2"
    else:
        return "-"
    t = base + s
    return "{:d}'{:02d}\"".format(int(t // 60), int(t % 60))


def transitions(labels, blocks):
    out = []
    for _, first, last in blocks:
        seg = labels[first:last + 1]
        for i in np.flatnonzero(seg[1:] != seg[:-1]) + 1:
            a, b = seg[i - 1], seg[i]
            kind = "team" if {a, b} == {"H", "A"} else ("dead" if b == "D" else "restart")
            out.append((first + int(i), a, b, kind))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", default=os.path.join(ROOT, "evaluation", "results", "model_v4.csv"))
    ap.add_argument("--n", type=int, default=20, help="total moments to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lead", type=float, default=3.0, help="seconds to start before the moment")
    ap.add_argument("--out", default=os.path.join(ROOT, "review", "transitions_review.csv"))
    args = ap.parse_args()

    labels = np.loadtxt(args.file, delimiter=",", skiprows=1, usecols=1, dtype="<U1")
    blocks = half_video_blocks(GAME)
    trans = transitions(labels, blocks)

    rng = np.random.default_rng(args.seed)
    share = {"team": 0.4, "dead": 0.3, "restart": 0.3}
    picked = []
    for kind, frac in share.items():
        pool = [t for t in trans if t[3] == kind]
        k = min(len(pool), max(1, round(args.n * frac)))
        picked += [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]
    picked.sort()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["no", "jenis", "frame", "putar_video_dari", "waktu_tepat_di_video",
                    "menit_pertandingan", "model_bilang", "verdict (benar/salah/ragu)", "catatan"])
        for i, (frame, a, b, kind) in enumerate(picked, 1):
            w.writerow([i, KINDS[kind], frame, video_time(max(0, frame - args.lead * FPS)),
                        video_time(frame), match_clock(frame, blocks),
                        "{} -> {}".format(NAMES[a], NAMES[b]), "", ""])

    # Local review player: opens in the user's own browser and plays the local video file. Nothing
    # leaves the machine.
    moments = [{"no": i, "kind": KINDS[kind], "frame": frame, "t": frame / FPS,
                "video_time": video_time(frame), "clock": match_clock(frame, blocks),
                "model": "{} -> {}".format(NAMES[a], NAMES[b])}
               for i, (frame, a, b, kind) in enumerate(picked, 1)]
    with open(os.path.join(HERE, "review_player_template.html"), encoding="utf-8") as fh:
        page = fh.read()
    key = "{}-{}-{}".format(os.path.splitext(os.path.basename(args.file))[0], args.seed, args.n)
    page = page.replace("__MOMENTS__", json.dumps(moments, ensure_ascii=False))
    page = page.replace("__KEY__", json.dumps(key))
    html_path = os.path.join(os.path.dirname(args.out), "review_player.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(page)

    counts = {k: sum(t[3] == k for t in trans) for k in KINDS}
    print("transitions in {}: {}".format(os.path.basename(args.file), counts))
    print("wrote {} moments to {}".format(len(picked), os.path.relpath(args.out, ROOT)))
    print("review player: {}  (open it in your own browser)\n".format(
        os.path.relpath(html_path, ROOT)))
    print("{:>3}  {:<32} {:>12} {:>9}  {}".format("no", "jenis", "video", "menit", "model"))
    for i, (frame, a, b, kind) in enumerate(picked, 1):
        print("{:>3}  {:<32} {:>12} {:>9}  {} -> {}".format(
            i, KINDS[kind], video_time(frame), match_clock(frame, blocks), NAMES[a], NAMES[b]))


if __name__ == "__main__":
    main()
