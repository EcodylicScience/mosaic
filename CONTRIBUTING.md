# Contributing to mosaic

Thanks for your interest in contributing.

## Scope

This repository contains backend tooling for behavioral analysis pipelines. Please open an issue before large changes so architecture, scope, and compatibility can be agreed first.

## Contribution Workflow

1. Fork the repository and create a focused branch.
2. Make your changes with tests and documentation updates where relevant.
3. Ensure the project installs and imports cleanly:
   - `pip install -e .`
   - `python -c "from mosaic.core.dataset import Dataset; print('OK')"`
4. Open a pull request with:
   - clear summary of the change
   - motivation and design notes
   - testing evidence (commands run and outcomes)

## Environment Notes

`pip install -e .` is enough: every dependency, `mosaic-media` included,
resolves from PyPI.

To work against an unreleased `mosaic-media`, clone it as a sibling and install
it editable *over* the released wheel:

```bash
pip install -e "../mosaic-media[io,cli]"
```

Order matters only in that the editable install wins whenever it is the later
one. CI does exactly this, so the suite runs against that repository's `main`
rather than its last release.

Prefer `uv pip install` over `uv sync`. `uv sync` installs the project without
extras and prunes anything it considers extraneous, silently undoing an extras
install.

**`uv lock` currently does not resolve at all**, on any platform:
`lightning-action` requires `nvidia-dali-cuda110` with no environment marker, and
PyPI serves that as an sdist only, so the resolver tries to build NVIDIA DALI from
source and fails. `uv.lock` is therefore stale and cannot be regenerated until
that is settled — by marker-gating the requirement, by dropping the extra from the
resolution, or by upstream publishing wheels. (Ultralytics used to be the blocker,
for a different reason; it no longer appears in mosaic's dependency graph at all.) Nothing in
the repository consumes the lock: every CI job installs with `uv pip install`,
and `scripts/gen_third_party_inventory.py` is the one reader, so `NOTICE`
regeneration is blocked with it.

## Code and Review Expectations

- Keep changes minimal and scoped to one concern per PR.
- Preserve backward compatibility unless a breaking change is explicitly discussed.
- Add or update tests for behavior changes.
- Update docs/notebooks if user-facing behavior changes.

## Contributor License Agreement

Two things are true at once, and both are intended. Your contribution enters this repository under the repository's own license, and separately you grant Ecodylic Science a license broad enough to also distribute the work under other terms. The second does not narrow the first: the AGPL grant to everyone who receives this repository is irrevocable, so nothing contributed here can later be withdrawn from the AGPL version, and a differently licensed distribution by Ecodylic Science is an additional channel rather than a replacement.

By submitting a contribution, you confirm that:
- you have the right to submit the contribution,
- the contribution is your original work (or you have required permissions),
- your contribution is offered to this project, and distributed by this project, under the GNU Affero General Public License v3.0 or any later version — the same license as the rest of the repository,
- you additionally grant Ecodylic Science a perpetual, worldwide, non-exclusive, royalty-free, irrevocable license to use, modify, sublicense, and distribute your contribution under any license terms, including terms other than the AGPL, so that Ecodylic Science may offer this project under a commercial license alongside the AGPL one,
- you retain copyright in your contribution; the grant above is a license, not an assignment.

## License

Contributions are provided under the AGPL-3.0-or-later license in `LICENSE`. The additional grant above lets Ecodylic Science also distribute this project under other terms; it changes neither the license under which you received this repository nor the license under which any contribution reaches a recipient of it.

Third-party components that mosaic invokes, optionally installs, or downloads weights from carry their own licenses, and some carry obligations the AGPL does not — keypoint-MoSeq prohibits commercial use outright, and TRex requires a paid license for company use. See [NOTICE](NOTICE).

## Security

Please do not open public issues for sensitive vulnerabilities. Report security concerns privately to the maintainers.
