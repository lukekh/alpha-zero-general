# Variable-value MVV-LVA capture ordering

`SearchConfig(mvv_lva_enabled=True)` enables most-valuable-victim,
least-valuable-attacker ordering in Minimax. `RustTeacher.analyze` accepts the
same option. It defaults to false, is included in search identity and native
reused-search configuration, and does not change evaluation or legal moves.

Capture keys use pre-move values from the existing variable-material formula.
The victim's value is computed using the opponent's army as its own army; the
attacker's value uses the moving player's army. Counts, prey/predator ratios
and balance therefore update after each capture. Highest victim value comes
first; equal victim values prefer the least valuable attacker. The final
existing action tie-break remains deterministic. With flat material all types
are 100, so these extra keys preserve the old move order.

Material value does not decide who wins an exchange under a cyclic capture rule,
so these keys are weak on their own. `see_ordering_enabled` adds an exchange key
that ranks above the victim/attacker keys; see [EXCHANGE.md](EXCHANGE.md). The two
are independent switches and either can be used without the other.

Immediate wins, the preferred TT/PV move, previous root scores and corner defence
retain priority. Enhanced ordering also retains its existing safe-capture and
escape groups. MVV-LVA orders captures within those tactical groups, ahead of
killer/history and distance tie-breaks. Both reference and compiled Python
paths implement the same priorities. Proof kernels keep their existing order;
MVV-LVA is not a proof or a pruning rule.

Completed unpruned Minimax scores remain unchanged, although tied actions,
work and elapsed time can differ. Under selective pruning or finite caps,
move-order changes can affect scores, chosen moves and which depth completes.
`SearchResult.ordering` and native `ordering` diagnostics count nodes ordered
and capture candidates ranked. Python charges its bounded extra ordering work
before sorting; native retains its existing node-based work units.

Native's newest search form extends the combined selective/evaluator request
with one `true/false` MVV-LVA flag before state. All earlier forms remain
accepted; the Python adapter uses an earlier form when this option is false.
No action encoding, game rule, static weight, or dataset label contract changes.

## Testing variable material at one twentieth scale

Use a variable-mode candidate with `count_weight=5`, retaining the other adopted
coefficients. Since normalized variable material is `army_value / 100`, this
is exactly equivalent to reducing the formula's BASE from 100 to 5 while
keeping the existing material coefficient at 100. Flat opponents retain
`count_weight=100`. Changing the global BASE in both the numerator and the
normalization would cancel the intended reduction, so the experiment uses the
existing coefficient explicitly.

Per-type explanation values still show the formula at BASE=100; multiply them
by `count_weight/100` for effective per-piece material contributions. Positive
uniform scaling leaves the MVV-LVA order unchanged. It does alter the balance
between material and the other terms, and the experimental futility allowance
already accounts for the material coefficient. This scale is a benchmark
candidate, not a new application default.
