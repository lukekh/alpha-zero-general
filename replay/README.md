# Off-policy replay of AlphaZero search

Re-run any search configuration over recorded network evaluations, at no
inference cost. This is the Dream-RSI replay-simulator idea applied to the
self-play loop: the network is frozen for a whole iteration, so `state ->
(policy, value)` is a pure function, and recording it once turns search-setting
comparison into an offline read.

`MCTS` needed no changes. It reaches the network only through `nnet.predict`,
so substituting a pool-backed object replays it.

## Why the pool, and not `MCTS.nodes_data`

`MCTS.nodes_data` already memoises per-state data, but its stored policy has had
Dirichlet noise applied and been renormalised in place, so replaying from it
would inherit the recording run's particular noise draw. The pool stores the
*raw* network output instead, keyed by a 16-byte BLAKE2b digest of the same
bytes MCTS uses for node identity -- a serialized Intransitive state is 6,804
bytes, far too large to key a large dictionary on.

Entries are stored sparsely over the legal-move mask. A network that masks
invalid actions leaves exact zeros outside it, which measured true for every
one of 877 recorded states, so a typical Intransitive position costs about 46
floats rather than 648. Any evaluation that does place mass outside the mask is
stored densely and counted as `mask_violations`.

## Exactness

Replaying the recording configuration must reproduce the recording exactly.
`python -m replay check` asserts it:

```
{ "agreement": 1.0, "miss_rate": 0.0, "policy_tv": 0.0, "exact": true }
```

This holds because the recorded search is deterministic: `prob_fullMCTS` is
pinned to 1 and every root uses `force_full_search`, so no RNG is consulted;
Dirichlet noise is off; and `universes` fixes the make-move seed. A pool
recorded with `--tree-reuse` cannot satisfy the check, because a recorded
decision then depends on statistics inherited from earlier plies that a
fresh-tree replay neither can nor should reproduce.

## What may be swept, and what may not

The action space is **compute allocation only**: `numMCTSSims`, `cpuct`, `fpu`,
`forced_playouts`, `universes`.

Dirichlet noise and the move-selection temperature are deliberately excluded.
They trade per-move quality for training-data diversity, and an objective scored
on per-move agreement would drive both to zero, score beautifully on replay, and
destroy training. Off-policy replay structurally cannot see that consequence,
because the pool was generated under the old distribution. Tune those online.

`prob_fullMCTS` and `ratio_fullMCTS` are also out of scope: they decide which
moves get a full search during self-play, which is a data-generation choice.
Read them off the quality-versus-simulations curve this tool produces instead.

## Misses are reported, never hidden

Replay is exact only while the alternative search stays inside the recorded
state set. A larger budget or a higher `cpuct` eventually steps outside it, and
every such step is counted. **A sweep row with a high miss rate is a partial
rerun, not a free evaluation.** Three policies:

| `--on-miss` | behaviour |
| --- | --- |
| `uniform` (default) | count the miss, continue from a uniform prior with value 0 |
| `strict` | raise, to prove a configuration stays inside the pool |
| `delegate` | call the real network, counting the cost that was not free |

## Splits

Games -- not positions -- are split into `select` and `confirm`, because plies
within a game correlate. Choosing a configuration by argmax over many variants
on one pool is repeated inspection, exactly what
`intransitive/evolution/README.md` already polices. Select on `select`, confirm
on `confirm`, and require an online confirmation iteration before adopting
anything.

## Commands

```sh
# 1. record a pool (and its reference decisions)
python -m replay record --game intransitive \
    --checkpoint checkpoints/<run>/best.pt --games 32 --sims 800 --out /tmp/pool

# 2. size it, and project a whole iteration
python -m replay measure --pool /tmp/pool

# 3. prove the replay is exact
python -m replay check --pool /tmp/pool

# 4. measure the machine's inference-batch latency curve
python -m replay bench --pool /tmp/pool --checkpoint checkpoints/<run>/best.pt \
    --out /tmp/pool/latency.json

# 5. sweep compute allocation against the recorded reference
python -m replay sweep --pool /tmp/pool \
    --grid numMCTSSims=100,200,400,800,1600 cpuct=1.0,1.25,1.5 \
    --latency /tmp/pool/latency.json --beta1 0.2

# 6. optional: an outside reference, then sweep against it
python -m replay label --pool /tmp/pool --engine rust --depth 6 \
    --node-limit 200000 --accept-depth 4
python -m replay sweep --pool /tmp/pool --reference teacher --beta1 0.2
```

`record` writes `pool-manifest.json`, `shard-*.pkl.z`, `positions.pkl.z` and
`positions-manifest.json`; every shard carries a SHA-256 that `load` verifies.

### Teacher labels

Agreement with the recorded search only says which configuration converges to
the recording's own opinion. That is the right question for compute allocation,
but it says nothing about whether the opinion is good, so `label` adds an
outside one. `--engine python` uses `intransitive.heuristics.AlphaBetaPlayer`
and always works; `--engine rust` is far faster but needs a binary matching the
Python client's wire format.

> The native binary lives at
> `intransitive/rust_teacher/target/release/intransitive-rust-teacher` and is a
> gitignored build artifact, so a checkout does not carry it and a stale one is
> not visible in `git status`. A binary older than the Python client rejects
> every request with `Expected search depth ms nodes radius weight proof_depth
> proof_nodes table_entries state_hex`; the fix is
> `cargo build --release --offline --manifest-path
> intransitive/rust_teacher/Cargo.toml`, or `--binary <path>` to a fresh build.
> Rebuilt 21 September 2026 (`d58b0e91`), validated by `cargo test --release`
> (37 passed) and `intransitive.tests.test_rust_teacher` (8 passed).

Labels are cheap on purpose: many shallow fixed-node labels beat a few deep
ones. `--accept-depth` keeps a partial iterative-deepening result that reached a
stated floor, rather than discarding it or accepting an unstated one. The depth
actually reached is recorded per position in `depth_histogram`.

## The parallelism term, and why there isn't one

Dream-RSI's objective is `quality - b1*cost + b2*parallelism`, where the bonus
rewards a policy that does its work in fewer, wider rounds. It needs that bonus
because it cannot price a hypothetical policy's wall clock: its probes are LLM
calls of variable cost. Here the network's batch-latency curve is a cheap,
stable property of the machine, so the proxy can be replaced by the measurement
-- and the measurement kills the term.

**First, batch width is not a search decision.** `predict_server` batches one
in-flight leaf per *concurrent game*, capped by `--parallel-inferences`. Each
worker's MCTS is strictly sequential -- one leaf, block, wait for the batch,
resume -- because this repo deliberately has no virtual loss. So batch width is
identical for every configuration a sweep can express, and `b2` cannot order
them.

**Second, making it a search decision would not pay.** The measured curve
(8-core Darwin, `flybrain-training-20260916/best.pt`) is flat:

| batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ms per position | 1.157 | 1.143 | 1.149 | 1.130 | 1.128 | 1.140 | 1.139 |
| speedup vs batch 1 | 1.00x | 1.01x | 1.01x | 1.02x | 1.03x | 1.02x | 1.02x |

Latency scales linearly with width, so gathering `k` leaves per call costs `k`
times as much and saves nothing. The mechanism is
`GenericNNetWrapper.export_and_load_onnx`, which pins the session to
`intra_op_num_threads = inter_op_num_threads = 1`: with one thread there is no
idle hardware for a wider batch to fill. That is a deliberate choice -- the
eight cores are spent running eight concurrent games, not one wide inference.

So the objective collapses to a single cost term. `sweep --latency` prices cost
in modelled wall clock, `ceil(evals / collection) * latency(collection)`, which
under this curve is proportional to evaluation count. At `--collection 1` the
model is exact, being today's search. Above 1 it is *optimistic*: it prices the
time of gathering leaves but not the decision quality lost by choosing them off
an un-backed-up tree, which needs a virtual-loss search to measure. The table
shows absolute `ms/move` for that reason -- on this curve every collection size
above 1 is a straight regression (11.25 -> 15.90 -> 36.46 ms at 12 simulations
for collections 1, 8, 32).

**Worth a separate look.** The flat curve is a configuration consequence, not a
hardware limit. The same model at four ORT threads costs 0.34 ms per position
at batch 8, against 1.13 ms in the production single-threaded path. Converting
that into self-play throughput means rebalancing worker count against session
threads for the same eight cores, so only an end-to-end measurement settles it
-- but it is a much larger number than anything `b2` was going to find.

## Metrics

| Column | Meaning |
| --- | --- |
| `agree` | top-1 match with the chosen reference, over every scored position |
| `confirm` | the same over held-out games; a gap means the split was overfitted |
| `TV` | total-variation distance from the *recorded* visit distribution |
| `evals` | network evaluations per move |
| `ms/move` | modelled wall clock, when `--latency` is given |
| `miss` | share served from outside the pool |
| `V` | `agreement - beta1 x (evals / reference evals)` |

## Not implemented

- **The parallelism term** -- measured and rejected, see below.
- **The self-improving outer loop.** Recording, replay and scoring come first;
  an automatic redeploy step is worth building only once the measurements show
  the search settings are worth moving.
- **Coach integration.** Nothing in `Coach.py` or `MCTS.py` was changed. The
  recorder plays its own games against a checkpoint, which keeps the training
  loop untouched while these measurements are still diagnostic.

## Verified on

Four games, 32 simulations, `checkpoints/flybrain-training-20260916/best.pt`:
877 unique states, 83 positions, 0 mask violations, 46.5 mean legal moves,
394 bytes per state, exact self-check. A simulation sweep behaved as designed --
agreement rising to 1.000 at the recording budget, and a budget above it
correctly flagged at a 37.7% miss rate. Those numbers demonstrate the machinery;
four games establish nothing about search settings.

Unit tests: `python -m unittest tests.test_replay` (28 tests, no checkpoint or
compiled rules engine needed).
