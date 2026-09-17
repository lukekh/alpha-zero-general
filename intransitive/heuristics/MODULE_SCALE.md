# Heuristic module scales

Example: the saved five-versus-three-piece endgame at seed 1643, state hash
`215c132ea32d242dc5cffde8c731432072b30e9fb24df12698e57f77c9e17981`.
All numbers favour player zero when positive. This is static evaluation with
proof disabled, at the adopted coefficients. See the exact
[saved evaluations](../benchmarks/variable-material/initial-evaluations.json).

| Module | Effective coefficient | Raw own-minus-opponent feature | Flat contribution | Variable contribution | Variable at 1/20 |
| --- | ---: | --- | ---: | ---: | ---: |
| Material | 100 (5 in scaled experiment) | 2 flat / 20.0968 normalized variable | +200.00 | +2009.68 | +100.48 |
| Piece advantage | 23.9671 | 4.6667 | +111.85 | +111.85 | +111.85 |
| Attacking position | 25.7145 | 1.7706 | +45.53 | +45.53 | +45.53 |
| Defensive position | 32.5643 | −0.2 | −6.51 | −6.51 | −6.51 |
| Overload / local pressure | 0, disabled | Not computed | 0 | 0 | 0 |
| **Total** | | | **+350.86** | **+2160.55** | **+251.35** |

The raw coefficients alone do not show module influence. Flat material is 100
per piece; the variable formula can value one piece at several times that.
Across the eight frozen endgames, variable material's weighted difference ranged
from −842.47 to +2009.68, advantage from −68.91 to +220.70, attack from −34.64 to
+45.53, and defence from −9.25 to +78.70. These are observed values on a small
endgame sample, not general bounds.

Ordinary static totals are clipped to ±10000. Proven wins/losses use approximately
±100000 (adjusted for mate distance) and replace the ordinary sum. `clear_run`
does not add an unproven positional bonus; it is zero in this example.

For the scaled variable experiment, use material coefficient 5 while the flat
opponent retains 100. This is equivalent to BASE=5 with the fixed normalization
of 100, and leaves all other terms unchanged. It is an experimental candidate,
not an adopted replacement for the defaults.
