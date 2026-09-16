# Native incremental evaluation evidence

See the [report](../../../rust_teacher/INCREMENTAL_EVALUATION.md) for interpretation and reproduction.

- `results.json.gz`: all 108 final pairs, configurations, input/binary hashes, timings and complete search results.
- `summary.json`: uncompressed summary and validation counts.
- `baseline.tar.gz`, `final-candidate.tar.gz`: the measured Rust source trees and Cargo files.
- `positions.tar.gz`: the six exact state-v2 inputs, including active history.
- `environment.json`: compiler version and source/binary identities.
- `final-rust-tests.log`, `final-release-tests.log`: nine passing native tests in each build mode.
- `final-python-tests.log`: eleven passing cross-language/backend test groups.
- `SHA256.json`: content hashes for the evidence files.

Both release binaries were built with Rust 1.74.1 and identical Cargo release settings.
The benchmark ran at low priority with existing training/generation jobs active.
No running worker was migrated to the new binary by this experiment.

Clean-branch validation also passed nine native release tests, six standalone
Python/native test groups, formatting and Clippy. `merge-python-tests.log` records
the standalone test run without the pending browser/dataset integrations.
