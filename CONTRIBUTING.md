# Contributing to behavior

Thanks for your interest in contributing.

## Scope

This repository contains backend tooling for behavioral analysis pipelines. Please open an issue before large changes so architecture, scope, and compatibility can be agreed first.

## Contribution Workflow

1. Fork the repository and create a focused branch.
2. Make your changes with tests and documentation updates where relevant.
3. Ensure the project installs and imports cleanly:
   - `pip install -e .`
   - `python -c "from behavior import Dataset; print('OK')"`
4. Open a pull request with:
   - clear summary of the change
   - motivation and design notes
   - testing evidence (commands run and outcomes)

## Code and Review Expectations

- Keep changes minimal and scoped to one concern per PR.
- Preserve backward compatibility unless a breaking change is explicitly discussed.
- Add or update tests for behavior changes.
- Update docs/notebooks if user-facing behavior changes.

## Contributor License Agreement

By submitting a contribution, you confirm that:
- you have the right to submit the contribution,
- the contribution is your original work (or you have required permissions),
- you grant Ecodylic Science a perpetual, worldwide, non-exclusive, royalty-free, irrevocable license to use, modify, and distribute your contribution under any license terms,
- you understand the contribution will be distributed under this repository's GNU Affero General Public License v3.0 (or any later version).

## License

Contributions are provided under the AGPL-3.0-or-later license in `LICENSE`.

## Security

Please do not open public issues for sensitive vulnerabilities. Report security concerns privately to the maintainers.
