# Compare Python and Rust minimax in the browser

Build the native engine, then start the normal play server:

```sh
cargo build --release --manifest-path intransitive/rust_teacher/Cargo.toml --offline
.venv/bin/python -m intransitive.play --port 8765
```

The opponent menu contains two independent choices:

- **Minimax · Python** (`alphabeta`): the existing `heuristics.AlphaBetaPlayer`.
- **Minimax · Rust** (`rust-minimax`): `rust_teacher.browser.RustMinimaxOpponent`,
  which invokes the native Rust executable through `RustTeacher`.

Select **Watch bots**, choose Python for Blue and Rust for Red, set the search
limits, then click **New game**. Reverse the colours for another game. Both
bots receive the same depth/time/numeric budget settings. You can also play
either engine yourself. `--opponent rust-minimax` selects Rust at startup.

Below the bot labels, each searched move shows its engine, completed depth,
elapsed seconds, nodes and score. **Last AI analysis** includes the budget unit,
stop reason, selection source and additional details. These diagnostics are
stored with each move, so stepping through history or reloading the page shows
the original engine's results. Copied game records identify both engines.

## Interpreting differences

Use core scoring and a modest fixed depth with generous time/work limits when
checking move or score agreement. Rust counts node visits; Python's work budget
also charges heuristic operations. Identical numeric work limits can therefore
stop the engines at different depths. Tied moves, ordering, table policies and
partial-search handling may also differ; a different move is not automatically
a scoring disagreement. See [native engine limitations](README.md).

Rust supports depths 1–32, default material weights and optional ring pressure.
Its Attack, Defence and Overload modules are unsupported, so those settings
must be off. Unsupported scoring configurations are rejected before changing
the current game. A shared `--ab-config` can enable supported pressure scoring
for both engines. Native PVS/ordering behavior is fixed by its implementation;
Python-only search optimization flags do not change the native engine.

The browser adapter starts a native process per move and closes it on success
or failure. It does not use the training adapter's process-wide signal handlers
and is safe to call from HTTP request threads. Rust elapsed time includes
process/transport overhead; `native_search_seconds` reports native search alone.
Each native move starts with a fresh table. Binary hashes identify the executable
used and detect rebuilding it during a game; start a new game after a rebuild.

There is no fallback to Python search. If the time or node budget expires before
Rust completes depth one, the browser uses the first legal move and explicitly
reports `legal_fallback`, depth zero and no score. Native execution errors are
reported as errors. Official game rules remain managed by the existing server;
both searches receive its modelling observation as before.

## Tests

```sh
.venv/bin/python -m unittest intransitive.tests.test_rust_browser \
  intransitive.tests.test_play intransitive.tests.test_record -v
node --test intransitive/tests/test_playback.cjs
```

Tests exercise actual Rust subprocesses in worker threads, prohibit calling
Python search from the Rust path, compare a completed-depth score, play both
engines together, preserve historical diagnostics, and check error/budget paths.
