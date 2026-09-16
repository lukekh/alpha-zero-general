# Opt-in heuristic scales (issue #53)

This is a configuration contract for Minimax/alpha–beta experiments, not an
optimizer, tournament, MCTS setting, or migration of a teacher or application
default. Issues #54–#56 own matches, evolution and held-out adoption respectively.

The opt-in [population optimizer](../evolution/README.md) now consumes this
contract through the paired match harness. It exports candidates for separate
held-out acceptance; it does not migrate these defaults.

The version is `intransitive-module-scales-v1`. The checked-in
[JSON schema](module-scales-v1.schema.json) uses JSON Schema Draft 2020-12.
`Genome.from_json` validates complete records without an extra dependency and
also rejects duplicate keys, booleans masquerading as numbers, NaN and infinity.
Unknown fields, unsupported backend genes and incomplete serialized genomes fail
before engines or games need to be started. `Genome.from_genes` is the explicit
partial-override API; missing genes take the effective defaults below.

| Gene | Python bounds | Effective default | SearchConfig mapping | Execution cost |
| --- | --- | --- | --- | --- |
| `advantage` | 0–100 | 25 | `advantage_weight` | Cached six-type material arithmetic after piece indexing |
| `attack` | 0–100 | 0 (off) | `attack_weight`, `attack_enabled` | Route maps, safe steps and capture opportunities |
| `defence` | 0–100 | 0 (off) | `defence_weight`, `defence_enabled` | Route/interception coverage and goal safety |
| `overload` | 0–100 | 0 (off) | `overload_weight`, `overload_enabled` | Coverage pairs, all legal first replies and renewed coverage; potentially expensive |
| `pressure` | 0–20 | 0 (off) | `pressure_weight`, `pressure_enabled` | Board indexing, two occupied-piece pair passes and cumulative rings; quadratic in occupied pieces |

All values are nonnegative magnitudes. The existing evaluator **subtracts**
overload. A positive optional gene enables its module; zero disables it and skips
its work. Dormant SearchConfig weights retain their legacy defaults (12/10/5/1),
so the default genome produces exactly `SearchConfig()`. They are not additional
dimensions. Advantage has no enable flag; zero removes its contribution.

Material count is fixed at 100. Genes are coefficients in those units, not
multipliers or arbitrary unnormalized weights. Integer/float representations and
negative zero normalize deterministically to floats and positive zero. No
rounding or log transform hides distinct candidates. `count_weight`, legacy
ignored `race_weight`, independent enable flags, internal advantage bonuses,
pressure radius/ring weights/defender discounts, terminal/proof values and draw
rules are deliberately excluded. v1 fixes radius 4 (9×9). Experiments with other
geometry or bonuses require a separate contract. Search depth, time/work limits,
proof budgets and aspiration settings are fixed run settings, not genes.

Fixing material removes the redundant uniform scale direction. It also preserves
the units of absolute aspiration windows (default 25), heuristic clipping (±10000)
and decisive scores (threshold 90000, mate 100000). Relative changes can still
increase aspiration retries or saturate ordinary scores. Terminal and proven
results override ordinary evaluation; clipping cannot create a proven win.

## Backend capabilities and reproducible manifests

Python supports all five genes. Rust supports **only `pressure`**, including zero.
A Rust genome cannot contain even a default-valued `advantage`, `attack`,
`defence` or `overload` gene. Its count/advantage coefficients stay 100/25, its
advantage bonuses stay 1/.5/.25, and attack/defence/overload stay disabled. No
native coefficient guard is bypassed and no native coefficients are changed.
Rust's ordering/PVS policy and work accounting differ from Python; equality of
fixed-depth scores is not equality of wall-time or node-budget protocols.
Native limits are validated before execution: depth 1–32, proof depth at most 2,
proof nodes at most 64, and at most one million table entries. Native time is
truncated to integer milliseconds by the existing adapter.
Python optimization flags in a SearchConfig are not native controls: the native
execution settings are recorded separately as `native_arguments` in its manifest.

```python
from intransitive.heuristics import AlphaBetaPlayer, SearchConfig
from intransitive.heuristics.tuning import Genome, saturation_report

candidate = Genome.from_genes({'advantage': 30, 'attack': 8, 'pressure': 2})
limits = SearchConfig(max_depth=3, time_limit=30, node_limit=10**8)
config = candidate.to_config(limits)
player = AlphaBetaPlayer(config=config)
manifest = candidate.manifest(limits, implementation_revision='FULL_GIT_COMMIT')
# Before matches, use the run's representative opening/midgame/endgame fixtures:
# preflight = saturation_report(candidate, fixture_states, base=limits)
# result = player.analyze(state)

# Native candidates are a separate, restricted contract:
native = Genome.from_genes({'pressure': 10}, backend='rust')
# with RustTeacher() as engine:
#     result = engine.analyze(state, **native.native_arguments(limits))
```

The base SearchConfig must have unmodified evaluation defaults; supplying other
weights, bonuses or radius fails rather than silently replacing them. A frozen
manifest contains the complete mapped config, backend contract version, exact
tunable dimensions and finite bounds, native arguments where applicable, score
limits, cache policy and caller-supplied implementation revision. Record the full
commit and a dirty-tree digest if applicable. `config_hash` identifies evaluation
semantics and backend; `manifest_hash` additionally covers revision and search
settings. Neither is a complete match/fitness cache key: #54/#55 must also include
opponent, full opening/history, rules, seeds and compute protocol.

Generate a reviewable manifest without starting an engine:

```sh
uv run --locked python -m intransitive.heuristics.tuning \
  intransitive/heuristics/configs/tuning-python-v1.json --revision "$(git rev-parse HEAD)"
```

Use `configs/tuning-rust-v1.json` for a pressure-only native manifest. These genome
files are loaded with `Genome.from_json`, not `SearchConfig.from_file`.

## Saturation, costs and cache safety

Every ordinary Python explanation now reports `raw_score`, `clipped` (magnitude
strictly above 10000) and `saturated` (at or above 10000). Search `module_calls`
reports `ordinary_evaluations`, `heuristic_clipped` and `heuristic_saturated`;
terminal/proof evaluations are excluded. This adds counters without changing
scores. Module timings and existing logical work counters remain available.

`manifest.saturation_possible` flags a conservative board-capacity bound. It uses
81 possible pieces, advantage at most 1.75 per piece, attack/defence/overload caps
3/4/2 and at most 1640 opposing pairs with pressure at most 2 each. This bound also
covers synthetic dense fixtures and can flag safe ordinary opening positions;
it is not a measured saturation rate. `saturation_report` evaluates a caller's
bounded fixture set, returns observed clipping/saturation rates and flags any
saturation or absence of ordinary samples. Exhausting the per-fixture budget
fails preflight. Its caller must freeze a representative fixture set and eligibility
policy before fitness; a range that merely saturates should not earn evidence of
improvement. For Rust candidates this preflight uses the parity-tested Python
formula; native searches do not expose leaf saturation counters. Native `inspect`
reports ordinary features even on terminal boards, so compare terminal status
separately from that diagnostic score.

Each Python search preparation recreates its evaluator/material/pressure caches.
The transposition table and hints are cleared on any `SearchConfig.identity()`
change; that identity includes every config field. Within an evaluator the
pressure cache stores unweighted raw totals, keyed by board and radius. Native
arguments request a fresh search (`reuse=False`); the native process also rebuilds
its search object when config changes. A candidate is immutable. Do not mutate an
engine's config during a running search or share an engine across concurrent
candidates.

The route and overload modules can cost much more than material or local
pressure. See [module measurements](MEASUREMENTS.md) and
[pressure measurements](../benchmarks/pressure/README.md). Future tournaments must
report fixed-depth and equal-wall-time outcomes separately and record per-module
cost; this contract makes no strength or throughput claim and starts no tuning run.

## Validation

```sh
cargo build --release --manifest-path intransitive/rust_teacher/Cargo.toml
cargo test --manifest-path intransitive/rust_teacher/Cargo.toml
uv run --locked python -m unittest \
  intransitive.tests.test_tuning intransitive.tests.test_heuristics \
  intransitive.tests.test_material intransitive.tests.test_pressure \
  intransitive.tests.test_rust_teacher -v
```

Constructed fixtures verify each gene's term, zero/enable behavior and overload
sign. Reused engines are compared with fresh engines after switching candidates.
Native candidate tests cover pressure features, material/score parity, captures,
Python make/unmake restoration, swapped goals, history-dependent terminal states,
fixed-depth valid/tied choices and changing candidates in one native process.
The existing native suite additionally tests 160 transition states, 100 certified
tactics and repetition/no-capture boundaries; Rust unit tests exercise native
make/unmake, incremental pressure/material restoration and cancellation.
