# Minimax depth 4: variable versus flat material

This follows the [depths 1–3 and MCTS comparison](../README.md), using the exact
same eight legal eight-piece endgames and the same adopted coefficients. Each
opening is played with colours swapped, for **16 main games**. Only the material
mode differs between opponents. The evaluator implementation remains unchanged;
this run records revision `e1ccdf7d5a15afb2fc0a5a91f9ae33b88ddcd72e`.

Results from **variable material's** perspective:

| Scoring | Wins | Losses | Unresolved | Failures | Unattempted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official | 6 | 6 | 3 | 1 | 0 |
| Official + consensus | 6 | 7 | 2 | 1 | 0 |

Unresolved means unfinished in the official row and inconclusive after
adjudication in the second row. Consensus does not rewrite the official game
outcomes. The official paired 95% interval is **[0.0, 1.0]**;
these are only eight opening clusters. The results do not by themselves
establish a reliable general strength improvement.

[Per-opening and per-colour results](comparison.json) retain all outcomes.
Played-search completed-depth counts across both opponents were `{4: 481, 3: 12, 2: 12, 1: 29}`;
stop reasons were `{'maximum_depth': 469, 'proven_result': 65}`. Earlier decisive proof can finish before
depth four; otherwise a played move must complete the requested depth.

## Paired retry with larger move limits

The opening whose primary game failed to complete depth four was retried in
both colours, with 60 seconds / 200 million work units per move. Game limits
remain 192 plies / 180 active seconds, with a separate 560-second pair wall
cap. The original one-hour supervisor deadline was retained. Primary records
are not replaced or pooled with this diagnostic.

Variable material recorded one official loss and one unfinished game. Consensus
awarded the unfinished game to variable material, giving **1 win and 1 loss**,
with no search failures or inconclusive games. The retry reproduced the first
11 moves of the failed game and completed depth-four searches beyond that point.

The retry supervisor took **6m21s**, with a sampled peak process-tree RSS of
**404.4 MiB**, nice 19 throughout and no priority anomalies. See
[retry details](retry-comparison.json) and [replay verification](retry-verification.json).


## Search limits

Both players request Minimax **depth 4**, with a larger per-move ceiling of
**30 seconds / 100 million logical work units**. Proof depth/tokens remain 2/64.
Games allow 192 plies or **180 active seconds**. This differs from the earlier
8-second / 20-million-work / 30-active-second caps, so comparisons with shallower
settings are not an experiment changing depth alone. Within this run, both
material modes receive identical limits.

A two-game, two-ply real-engine smoke precedes the main comparison. It uses the
same depth/per-move limits, a 60-active-second game ceiling and a 180-second wall
cap. The main comparison has at most 3200 wall seconds, reduced if necessary to
leave room for audit before 3400 seconds from study start. The outer supervisor
has a 3500-second deadline plus bounded cleanup within the authorized hour.
Game counts and per-move work caps also bound total reserved work; exact ceilings
are saved in each `budget.json`.

Only completed requested searches or earlier proved results may supply played
moves. A depth/work/time failure remains excluded rather than being credited
as a win. For otherwise unfinished games, the user's separate consensus rule
compares both competing static evaluations from the same player's perspective:
matching strict nonzero signs adjudicate a winner; ties or disagreement remain
inconclusive. Official outcomes and original journals are retained.

This is a Python strength comparison. The variable-material implementation in
Rust was parity-tested in the preceding change. No coefficients are retuned and
no application defaults change. These eight endgames are a small, deliberately
reused corpus; they do not establish opening/midgame performance or an independent
held-out gain. Do not pool results with other depths as independent samples.

## Measurements and verification

The supervised smoke, comparison, replay audit and consensus pass took
**22m58.86s** (1378.86 seconds), within the one-hour cap.
Observed priorities were `[19]`, with **0 priority anomalies**.
BLAS/OMP/MKL thread counts were one. Three-second sampling measured a peak
summed descendant RSS of **402.7 MiB**, at most
4 processes. These are shared-machine measurements. No observed
experiment descendants remained at cleanup.

All **18 records / 538 played moves** passed replay and exact report
reconstruction. Accounting totals including smoke: consensus labels
`{'consensus': 3, 'excluded': 1, 'inconclusive': 2, 'official': 12}`; adjudicator statuses `{'complete': 10}`. Smoke results
are not counted as strength evidence.


The retry added **2 records / 64 moves**, all verified, bringing the total to
**20 records / 602 moves**. Both phases and final cleanup finished by
**31m06.83s from the original launch**, within the authorized hour. The final
cleanup check found none of either experiment supervisor’s observed descendants
still present.

## Evidence

- [Frozen design](design.json), [comparison](comparison.json)
- [Consensus score pairs](consensus.json), [replay audit](verification.json)
- [Search completion](search-completion.json)
- [Measurements](measurements.json), [cleanup](cleanup.json)
- [Full archive](runs.tar.gz), [SHA-256](runs.sha256)


The full archive includes the exact positions, scripts, complete manifests,
journals, reports, score pairs and monitor logs. To repeat the run, use its
`study.py` and `supervise.py` at the recorded implementation, changing the
original `/tmp/issue55-variable-depth4-20260917` output root as needed, with an
explicit bounded resource allocation. Replaying saved games starts no engines.
