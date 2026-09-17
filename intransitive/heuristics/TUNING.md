# Signed heuristic scales (v2)

The v2 genome supports signed coefficients and tunable material for Python
Minimax and heuristic-value MCTS experiments. At the user's explicit request,
the defaults now adopt the [measured endgame candidate](../benchmarks/evolution/hour-search-20260917/README.md)
in both Python and Rust. This adoption does not imply that separate held-out
acceptance was performed. Future optimizer exports remain provisional.

| Gene | Bounds | Default | SearchConfig field |
| --- | --- | ---: | --- |
| `material` | −100 to 100 | 100 | `count_weight` |
| `advantage` | −100 to 100 | 23.967050360966205 | `advantage_weight` |
| `attack` | −100 to 100 | 25.714516982666414 | `attack_weight`, `attack_enabled` |
| `defence` | −100 to 100 | 32.5643023919054 | `defence_weight`, `defence_enabled` |
| `overload` | −100 to 100 | 0 | `overload_weight`, `overload_enabled` |
| `pressure` | −100 to 100 | 0 | `pressure_weight`, `pressure_enabled` |

Negative values reverse the contribution of a feature. The evaluator subtracts
overload, so a negative overload coefficient rewards that feature. Any nonzero
optional coefficient enables its module; zero disables its computation. Material
and advantage have no independent enable flag. Coefficients are not normalized
to sum to a constant or divided by material. The search therefore includes a
uniform scale direction, which can affect clipping, absolute search windows and
MCTS value calibration.

The version is `intransitive-module-scales-v2`; the checked-in
[schema](module-scales-v2.schema.json) and `Genome.from_json` require complete
records. Runtime validation rejects duplicate keys, unknown or unsupported
genes, booleans, nonfinite numbers and out-of-range values. Partial overrides
use `Genome.from_genes`; omitted genes take the defaults above. Negative zero is
canonicalized. Historical v1 schemas and genomes remain as archived examples;
use their original repository revision to reproduce them. They are not silently
reinterpreted using the new defaults or search space.

## Backends and fixed settings

Python supports all six genes. Rust supports material, advantage, attack,
defence and pressure. Its new static occupied-board route/interception features
match Python; overload remains unsupported and cannot appear in a Rust genome.
The adopted defaults have overload disabled in both backends.

Internal advantage bonuses (1/.5/.25), pressure geometry (radius four in the
genome), terminal/proof scores, draw rules and ordinary clipping remain fixed.
Search depth, time/work caps and MCTS settings are run settings, not genes. Rust
keeps its native ordering/PVS and visit accounting; Python counts broader
logical work. Equal completed-depth scores do not imply equal work or runtime.

`Genome.native_arguments` forwards every supported coefficient, uses a fresh
native search and validates native limits: depth 1–32, proof depth at most two,
64 proof nodes and one million table entries. The Rust search configuration
identity includes all coefficients, so changing weights also resets a reused
transposition table. Explicit old weights remain available through the adapter;
existing binaries and already-running training jobs are not hot-swapped.

## MCTS interpretation

The tournament's `mcts` protocol uses the repository's PUCT MCTS with uniform
legal priors and a heuristic leaf value `tanh(score / value_scale)` (default
scale 400). No neural model, rollout or Dirichlet noise is used. Each move gets a
fresh tree. Simulation count, exploration constant, value scale, proof limits,
time and work ceilings are frozen into the manifest and cache key. Visits and
heuristic/proof work share a cooperative budget.

MCTS records completed simulations and maximum observed tree depth separately
from Minimax's completed depth. An incomplete requested simulation count fails
the fixed-simulation match protocol. Different depths, simulation budgets and
search methods require separate experiments and leaderboards.

## Saturation, bounds and reproducibility

Ordinary scores are clipped at ±10,000; decisive scores remain outside that
range. Terminal and proven results override the heuristic even with negative
material. The conservative absolute bound uses the **absolute value** of every
coefficient: board capacity 81, advantage at most 1.75 per piece, route caps
3/4/2 and pressure bound 3,280. Saturation preflights measure raw scores on frozen
search fixtures. Truncated preflights or saturated fixtures cannot qualify a
candidate. Rust preflight uses the parity-tested Python formula.

Manifests contain complete effective configs, bounds, source/runtime identities
and search settings. Python cache identity includes the entire config. The
match harness also fingerprints the root MCTS implementation. No cache record
from the old defaults, old genome version or another search protocol is reused.

```python
from intransitive.heuristics.tuning import Genome

candidate = Genome.from_genes({'material': -50, 'attack': -20})
print(candidate.to_json())
print(candidate.manifest(implementation_revision='record-the-exact-revision'))
```

Use [Python v2](configs/tuning-python-v2.json) or
[Rust v2](configs/tuning-rust-v2.json) with `Genome.from_json`. These genome files
are not `SearchConfig.from_file` presets. The [optimizer contract](../evolution/README.md)
describes mutation, fitness, budget reservations and resume behavior.

The [variable-material mode](VARIABLE_MATERIAL.md) is a fixed, per-candidate
comparison option outside the v2 genome. V2 retains flat material; attempting to
use a variable-mode base with a genome is rejected. The paired tournament API
supports comparing the two modes with identical signed coefficients.
