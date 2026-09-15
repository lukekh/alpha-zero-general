# Flybrain opponent

`flybrain-wired.npz` contains neural responses reproduced from
[charbelkassab/flybrain-intransitive](https://github.com/charbelkassab/flybrain-intransitive),
commit `f22c07ad64a78b9e1423c376dbc289380ef6f6fc`.
It implements the upstream **fly wiring, zero training** player.

## Play our checkpoint against it

From the repository root:

```sh
.venv/bin/python pit.py intransitive \
  checkpoints/supervised-minimax5-20260915-095722/best.pt flybrain \
  --num-games 2 --numMCTSSims 32 --flybrain-seed 0
```

Arena alternates colours. `--numMCTSSims` controls our model's search budget;
Flybrain scores each legal move without search. Use `--flybrain-bank PATH` for
an alternative response bank with compatible metadata.

Human play is also available:

```sh
.venv/bin/python -m intransitive.play --opponent flybrain --port 8767
```

Select **Fly connectome** in the browser opponent menu. Our engine owns legal
moves, captures, canonical player labels, history and termination. Official
matches do not import the upstream game's assumed 200-ply stagnation draw.
Our neural model's internal search retains its existing modelling draw rules.

## What the bank contains

The complete MaleCNS model was built locally: 165,122 neurons, 25,563,197 raw
connections, and 24,000,270 connections after upstream stabilization. The
unmodified upstream `experiment.record_bank` ran all 32 binary feature patterns
12 times each, with brain seed 1 and 80 ms of simulation per response.

The five features are goal, capture, progress, danger and base threat.
`wired[pattern, repeat]` stores approach-neuron spikes minus escape-neuron
spikes. The adapter samples repeats **6–11** independently for every legal move,
then chooses uniformly among maximum values (within 1e-9), exactly as upstream
`experiment.brain_player(..., 'wired')` does. Candidates are enumerated in our
action order, so a seed does not reproduce the upstream game's entire trace,
but its sampling and selection distributions are preserved.

This is the upstream experiment's cached-response player. It does not run
165,122 neurons during each game, average away the stochastic responses, use
the fitted linear readout, or substitute the reference greedy heuristic.
Our `reference_greedy.features` already implements the upstream feature
semantics and handles both canonical player perspectives.

The compressed bank is about 2.3 KB; playing needs no new dependencies. Its
embedded JSON metadata records upstream source hashes, raw-data hashes,
simulator settings and build dependencies. It is loaded with pickle disabled.

Bank SHA-256:
`80b8aaf979bb9d3077894e3c1178b6c368708182ee9d452a025c1846eb3d01c0`

## Rebuild from the actual connectome

```sh
.venv/bin/python -m intransitive.flybrain_prepare
```

This clones and verifies the pinned upstream revision under the ignored
`intransitive/data/flybrain-intransitive/`, installs the build dependencies in
that checkout's own virtual environment, downloads approximately 1.1 GB of
Janelia data, builds the sparse connectome and simulates the response bank.
Allow several GB of memory during building. Nothing is trained. Existing
downloads are reused only when their sizes match the remote objects; new
downloads and bank writes use temporary files before publication.

To preserve the bundled bank while rebuilding:

```sh
.venv/bin/python -m intransitive.flybrain_prepare \
  --output intransitive/data/rebuilt-flybrain.npz
```

The simulator values should reproduce with the recorded dependencies; archive
bytes can differ because ZIP timestamps and environment metadata may differ.

## Verification (2026-09-16)

30 adapter, feature, browser and official-rule tests passed. The optional source
parity test uses the downloaded upstream selection functions unchanged and
compares all legal move features plus seeded move choices over 100 reached
positions, covering both colours. Fixture tests also check held-out response
sampling, invalid banks, state immutability, terminal rejection and pit/browser
integration.

All five reproduced single-feature averages exactly match upstream
`results/scoreboard.json`: goal 6.666667, capture 6.166667, progress 2.916667,
danger -115.916667 and base threat -196.916667 (rounded here).

Two initial official-rule games using our saved supervised checkpoint and 32
MCTS simulations both ended in corner wins for our model: Blue in 75 plies,
Red in 92 plies. Both action sequences were replay-verified. This is an
integration smoke test, not a strength estimate. Local checkpoint hashes,
move lists and results are in `checkpoints/flybrain-integration-20260916/`.

## Attribution and licenses

- Simulator and experiment: Copyright (c) 2026 charbelkassab, MIT; see
  [FLYBRAIN-LICENSE.txt](FLYBRAIN-LICENSE.txt) and the linked upstream repository.
- Connectome: **MaleCNS v1.0**, Google Research & HHMI Janelia FlyEM,
  [male-cns.janelia.org](https://male-cns.janelia.org/),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  This bank is a derived artifact: the raw connectivity is signed, stabilized,
  simulated and reduced to approach-minus-escape spike counts using upstream's
  implementation. The source URLs and SHA-256 hashes are recorded in the bank.
- Game: Intransitive by meaf; [game site](https://meaf.us/rps2/).
