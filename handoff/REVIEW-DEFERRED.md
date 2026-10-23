# Findings the whole-branch review deferred or rejected
Deferred = real, left for a later pass. Rejected = checked and dropped. Source: six-lens review of S1-S3, triaged 2026-10-23.

## Deferred
- **planner-4**: Real hole (a bound run fitted on a held-out group passes the guard) but closing it needs a design choice about reading a bound run's recorded fit scope (fit_scope.json / index rows) and what the bind confirmation should say; medium risk, new behaviour. Record as follow-up before S4/S5; the 'bind' confirmation is the interim control.
- **planner-7**: Documented design: a positive cv_gap_frames covering the known reach silences the undeclared-reach caution (stated in the S3 summary, graph/CLAUDE.md and reach_findings docstring). Tightening it is a policy change; the better fix is declaring reach for the common undeclared features (pair-position, pair-egocentric, ...), which is feature work. Follow-up.
- **quality-7**: Dedup of the cached-bundle cv_unit check and bundle block keys across xgboost and sequence-classifier: refactor of two classifiers with no defect; deferred.
- **quality-8**: Three block-lookup implementations: medium-risk refactor of fold dealing code that is golden-tested; the reviewer's disagreement for a frame outside every block is not shown to be reachable. Deferred.
- **quality-9**: Shared out-of-fold gathering loop for tune-threshold and probability-ensemble: medium-risk refactor, the two already differ on purpose (NaN handling). Deferred.
- **quality-10**: verify._build re-implements run_feature's preamble: the run-id equality check guards identity; making three run.py helpers public is churn and the larger extraction is high risk. Follow-up if verify drifts.
- **quality-13**: Splitting the 40-line stacked branch in _input_coverage_problems is taste with a regression risk in the planner; deferred.
- **quality-15**: Moving the unresolved-model refusal of sequence-classifier/lightning-action from fit() to named_model_file in load_state changes exception types and tests (medium risk) for a wording/consistency gain. classifiers-4 aligns xgboost's message with the current fit()-time wording instead.
- **tests-3**: Test-time reduction (module-scoped fixtures for tune-threshold and ensemble pipeline files). Real concern since the test job hit its 20-minute timeout on main, but medium-risk restructuring of many tests and unrelated to correctness; take it as a separate pass with CI timings.
- **tests-4**: Module-scoped study fixture copied per test in verify/score tests: same CI-time concern and risk as tests-3; separate pass.
- **tests-5**: Sharing the CalMS21 recipe front half across ten test files is a broad, behaviour-neutral refactor; low value for a tidy pass and invites merge conflicts with main.
- **docs-6**: Merge/rebase conflicts with main (counts 53 features and 33 commands, CHANGELOG, generated reference, golden JSON) are a merge-time task; nothing to change on the branch now. Recorded for the merge: keep both CHANGELOG sides, set README to 53 features / 33 commands, regenerate docs/reference with gen_docs_reference.py --write, rerun tests/meta/test_docs_assets.py and the golden run-id test.

## Rejected
- **planner-8**: The changed caution text for an unconverted dataset replaces main's nonsense 'covers all 0 entries'; fingerprint and plan are unchanged. An improvement, not a regression; the 'plans as it did' test covers the converted case.
- **quality-19**: Merged into surface-7 (clear error for a mistyped --answers path). Dropping bare-path acceptance would break SKILL/guide examples; unifying every option on bare paths and sharing the md5 helper is taste and a CLI behaviour change.
- **tests-6**: Taste: about 200 skipped parametrized cases are noise, not a defect, and re-parametrizing 6 conformance tests over computed subsets is churn for no behaviour change.
- **tests-8**: About 13 s and a duplicated loop; no failure, and merging two subprocess programs makes the test harder to read.
- **surface-11 (remove half)**: splits remove exiting 1 for a role-less group is a deliberate typo check; only the set warning and de-duplication are accepted.
