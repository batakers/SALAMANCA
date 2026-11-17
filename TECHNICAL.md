# Technical document — Ball possession by HMM segmentation

*More than a Hack 2027, Phase 1 — team **SALAMANCA**. Match: FC Barcelona (H) vs Athletic Club (A).
Code: https://github.com/batakers/SALAMANCA*

## 1. Approach

We treat the task as **segmentation**, not per-frame classification. The ranking metric (`spellF1`)
rewards recovering each possession as one interval, and the challenge's own definition says
possession changes only on *controlled* possession. A model that decides every frame independently
fails on both: it flickers, and it hands the ball over on every touch.

We use a three-state model — `H`, `A`, `D` — and find the single best label sequence with
**Viterbi**:

- every frame contributes a **score** for each state (section 4);
- every **change of state costs** a fixed penalty: `4.5` for H↔A, `16` for alive↔dead (paid twice
  per stoppage).

The best path maximises total score minus switching costs. By construction this gives:

- **no flicker** — a change has to be paid for;
- **controlled possession** — a touch, a deflection or a lost aerial duel produces a few frames of
  opposing evidence, less than the cost of switching away *and back*, so it creates no possession;
- **sharp timing** — Viterbi places a switch where the new evidence starts, without the lag of a
  smoothing window.

The pipeline has **two passes**:

1. **Ball-based pass.** Team evidence from who is on the ball, dead-ball evidence from the ball and
   the players' activity.
2. **Team-shape pass.** A classifier learns, from the 22 players' positions and movement only,
   which team is in possession. It is trained by **self-training** on pass-1 labels where the ball
   is tracked (no human labels), cross-fitted by half, and its output becomes additional team
   evidence. This is what tells the teams apart while the ball is not tracked.

Why this rather than a supervised model: the rules forbid labels from this game, we had no labelled
data from other games, and the possession definition is largely a *temporal* rule that switching
costs express directly. Every parameter has a physical meaning.

## 2. Assets used, and training data

| asset | used | how |
|---|---|---|
| Tactical tracking (ATD + EPTS metadata) | yes, main input | ball and players per frame; each slot's team from the metadata |
| Smart Tagging | yes, lightly | half-opening kick-off times (dead ball before them); a small per-frame nudge from phase tags; with that nudge off, a label-free check (section 6) |
| Homography | no | tracking positions are already in pitch coordinates |
| Tactical video | debugging only | watched locally to understand failure modes (section 6); no label, frame or constant comes from it |
| Broadcast video | no | not needed for this approach; a possible next step (section 8) |
| Public 5-minute ground truth | verification only | consulted after each configuration was fixed (section 7) |

**Training data: none.** No labelled dataset of any kind was used — not from this game, not from
any other. The only learned component, the team-shape classifier, is trained on the model's own
first-pass output (self-training on tracking data); no human produced any of its targets.

## 3. Key finding about the data: the missing ball is mostly a stoppage

The ball is missing on 41% of graded frames, but almost never briefly: **75% of missing frames are
in runs longer than 15 s**, and there are 131 runs longer than 5 s — about the number of stoppages
in a match. Inside these runs mean player speed drops (1.9 → 1.2 m/s), and in 86% of them the ball
reappears already moving, i.e. at the restart kick. Only about 19% of them look like live play.

So the tracker generally drops the ball for the whole of a stoppage. We use a long missing run as
**evidence of a dead ball** (not as proof: high player activity overrides it), and its edges then
mark when play stopped and restarted. Short-gap interpolation, which we planned first, would have
done nothing: there are almost no short gaps.

## 4. Features and evidence

Positions in metres (pitch 105.3 × 68.0 m); ATD frame `i` is video frame `i-1`.

**Controlled possession (main scores for `H` and `A`).** A direct implementation of the
challenge's definition. A frame is a *control frame* for a team when one of its players is within
1.0 m of the ball, **no opponent is within 1.5 m**, and the ball is slower than 6 m/s (a dribble,
not a pass travelling past). Control frames only count inside a **run of at least 5 frames
(0.2 s)**, after bridging up to 2 frames of tracking jitter — a touch, a header or a deflection
lasts 1–3 frames and earns nothing. Each frame of a kept run scores +1.0 for that team.

**Soft closeness (secondary).** For each team, `exp(-½ (d / 1.5 m)²)` of its nearest player's
distance to the ball, damped by `1 / (1 + (v/8)⁴)` for ball speed `v`, weighted 0.3. It keeps some
evidence where control is not exclusive. When both teams are close (a contested ball) the scores
cancel and the current possessor keeps it, as the rules require. Only frames with the ball tracked
and inside the pitch contribute.

**Team shape (scores for `H` and `A`, pass 2).** 27 features per frame: for each team the centroid,
spread (std of x and y), 10th/90th percentile of x (defensive line, most advanced players),
centroid velocity and mean speed, plus home-minus-away differences. x is flipped in the second half
so each team attacks the same way in both halves. A gradient-boosted classifier
(scikit-learn `HistGradientBoostingClassifier`, fixed seed) predicts P(home in possession). Trained
on the *other* half only, it agrees with the ball-based labels on 77% of held-out frames. It adds
`w·(P − 0.5)` to `H` and subtracts it from `A`, with `w = 0.1` where the ball is tracked and
`w = 0.3` where it is not.

**Dead ball (score for `D`; positive = looks dead, negative = looks alive).**
- ball tracked more than 0.3 m beyond a touch or goal line: +1.0 per frame;
- ball lost while last seen within 0.5 m of a line or beyond it (carried up to 20 s): +0.5;
- ball missing in a run of 5 s or more: +0.3 (section 3);
- low collective activity: mean player speed over ±1 s below 1.4 m/s pushes towards `D`
  (saturating at ±0.35 per frame) — during stoppages players walk;
- ball nearly stationary (< 0.7 m/s over ±0.5 s): +0.3 — a ball being placed for a restart;
- ball faster than 5 m/s inside the pitch: −0.3 — a ball in flight is in play.

**Smart Tagging.**
- attacking phases (`BUILD UP`, `PROGRESSION`, `CREATING CHANCES`, `ATTACKING TRANSITION`,
  `BALL IN FINAL THIRD`, `BALL IN THE BOX`) add +0.05 per frame to the named team; defending phases
  and `DEFENSIVE TRANSITION` add it to the other team. Deliberately small: the tags are automatic
  and their intervals are phases, not possessions;
- before a half's `Kick Off (Start)` tag the ball is not in play: +2.0 to `D`.

Frames outside the two graded halves are written `X`.

## 5. Possession rules implemented

- **Change of possession**: exclusive control by the other team, sustained for at least 0.2 s and
  enough to pay the switching cost; or, with the ball untracked, a sustained team-shape signal.
- **Loose / contested balls** stay with the last team in possession (no net evidence).
- **Touches, deflections, aerial duels** without follow-up control do not switch.
- **Restarts**: after a stoppage the state is decided by the team that controls the ball once play
  resumes; dead-ball ownership is never predicted — the label is `D`.

## 6. How variants were chosen — without the ground truth

The rules forbid tuning on the public 5 minutes, so every design choice was made on **label-free
proxies** (`tools/proxy_eval.py`, `tools/matched_eval.py`), built only from automatic data and our
own predictions:

- **team agreement** — on frames inside Smart Tagging phases that name a possessing team, how often
  our H/A label agrees (per frame, and one vote per phase). Smart Tagging is algorithmic, not human
  labelled. Measured with the model's own tag prior switched off, so it is not circular;
- **restart recall** — share of Smart Tagging set-piece windows containing a predicted
  dead→alive transition (the model does not read these windows);
- **possession count, mean duration, dead-ball share** — compared with football norms (mean
  possession ~12–15 s, ball out of play ~40–45% of match time).

Because team agreement favours fewer switches (phases are long), variants were compared at a
**matched possession count** (the H↔A cost re-solved per variant to give ~280 possessions).

| step (matched at ~280 possessions) | team agree | phase agree | restarts | D% |
|---|---|---|---|---|
| organisers' `baseline_smooth50` (669 poss., unmatched) | 0.701 | 0.753 | 0.81 | 32 |
| ball-based model (v2; 270 poss., unmatched) | 0.742 | 0.753 | 0.88 | 39 |
| + long missing run as dead-ball evidence | 0.766 | 0.780 | 0.91 | 43.5 |
| + team-shape evidence (v3; 285 poss.) | 0.799 | 0.823 | 0.91 | 42 |
| + exclusive, sustained control runs (final, v4; 273 poss.) | **0.814** | **0.851** | 0.91 | 42 |

The last step was compared on the full two-pass pipeline. It wins at a similar possession count, and
also at a *higher* count (295 possessions: 0.806 / 0.830), so the gain is not the proxy's preference
for fewer switches.

Tried and not kept: a tighter soft-closeness radius (1.0 m) or fast-ball threshold (5 m/s) helped
slightly on their own but added nothing once team shape was in; heavier gap or team-shape weights
lowered restart recall; control runs as the *only* team evidence (cannot produce more than ~220
possessions even with a near-zero switching cost, far fewer than the frequent changes of possession
football shows); Smart Tagging `GOALS` intervals as dead-ball priors (they span the whole attack and
celebration, marking live play dead).

The final model predicts 260 possessions of mean 14.2 s and 41.9% dead ball on graded frames.

### Watching the video (debugging only)

To understand failure modes we watched 20 randomly sampled predicted transitions of v3 in the
tactical video (`tools/transitions_for_review.py`, a local player; the video never left the
machine). The main finding was qualitative: most wrong changes of possession were touches, duels,
deflections or a ball merely passing near a player, and timing errors went both ways (no
systematic offset). That motivated the exclusive, sustained control runs above, which come straight
from the challenge's definition of controlled possession. **No verdict from this review was used as
a label, to select a variant, or to set a constant tied to any moment of the game**; the review
sheet is gitignored and never read by the pipeline. The variant was selected on the label-free
proxies only.

## 7. Verification

- `validate_submission.py`: valid, 199,319 rows.
- Determinism: two independent runs give a byte-identical file
  (SHA-256 `3c5d190d…2edffab71f7d86`).
- Public 5-minute window, consulted **only after** each configuration was fixed, as a check that
  the pipeline is aligned and behaves sensibly:

  | | macroF1 | spellF1 | transF1 ±1 s | possessions (truth 17) |
  |---|---|---|---|---|
  | final (v4) | 0.745 | 0.471 | 0.256 | 17 |
  | v3 | 0.741 | 0.538 | 0.216 | 15 |
  | ball-based only (v2) | 0.697 | 0.424 | 0.158 | 17 |
  | organisers' `baseline_smooth50` | 0.621 | 0.326 | 0.082 | 26 |

  v4 versus v3 is mixed here: better per-frame, on transition timing and on possession count,
  lower on spellF1 by about one possession's worth. The scorer reports gaps below ~0.35 on this
  window as noise, and the organisers show that choosing on it picks the wrong variant, so we kept
  the variant chosen beforehand on the full-match proxies.

No hand labels, no hand corrections, and no human-produced data about this game were used. The
match data was never sent to any hosted service.

## 8. Limitations and what we would do next

- **Self-training bias**: the team-shape classifier learns from our own labels, so it inherits
  their systematic errors; Smart Tagging, used as a proxy, may share cues with it.
- **Transition timing** is still the weakest part (about one transition in four within ±1 s on the
  public window). Next: place each change at the first frame of the new team's control run rather
  than where the cumulative evidence crosses the switching cost.
- **Live play with the ball untracked** (~19% of long gaps): team evidence there is only team shape.
  Next: estimate the ball position from player convergence and velocities.
- **Short stoppages** (quick throw-ins and free kicks) can still be missed, merging two possessions of
  the same team. Next: a restart detector (stationary ball near a line or spot, then a sharp kick).
- **Goalkeeper holding the ball** can look like a stoppage; the metadata's `position_type` could
  identify keepers.
- **Fixed switching costs**: a duration-aware model (HSMM) or costs depending on pitch zone would
  match real possession dynamics better.
- **Broadcast feed unused**: aligning it could recover ball positions the tactical tracking misses.

## 9. Reproduction

```bash
pip install -r requirements.txt     # numpy, pandas, scikit-learn (pinned)
# data in data/tactical/ (or MTAH_DATA_DIR)
python make_submission.py --team SALAMANCA
python evaluation/validate_submission.py evaluation/results/SALAMANCA.csv
python tools/proxy_eval.py          # optional: the label-free checks
```

Python 3.11+, CPU only, under two minutes on a laptop. Expected SHA-256 of `SALAMANCA.csv`:
`3c5d190d093f942fec3bbdd186394158bf550831130e61b7fe2edffab71f7d86`.
