# Work with an AI coding agent

Run this in your project directory to give an AI coding agent the mosaic guide:

```bash
mosaic agent install
```

The guide covers the dataset, the CLI, the Python entry points and the mistakes to
avoid. It ships with mosaic, and an agent reads it from your installation, which keeps
it matched to the version you have. Its source is
[`src/mosaic/agent/SKILL.md`](https://github.com/EcodylicScience/mosaic/blob/main/src/mosaic/agent/SKILL.md).

## Files the install writes

`mosaic agent install [directory]` works in `directory`, or in the current directory
when you name none, and writes two things:

- `.claude/skills/mosaic/SKILL.md`, a short skill that tells the agent to run
  `mosaic agent guide` and follow the guide. It names your `mosaic` executable by
  its full path, which works in a shell where your environment is not active.
- A block of two lines in `CLAUDE.md` and `AGENTS.md`, pointing at the same command.
  Each file that exists gets the block, and `CLAUDE.md` is created when neither does.
  `--no-memory` skips this step.

The command prints each file it wrote. Running it again replaces the skill and the
block and adds nothing beside them. Run it again after you move or rebuild your mosaic
environment, which changes the path of the executable.

## Removal

Delete the `.claude/skills/mosaic/` directory, and the lines from
`<!-- mosaic-agent-guide:begin -->` to `<!-- mosaic-agent-guide:end -->` in
`CLAUDE.md` and `AGENTS.md`.

## Tools for an MCP client

`mosaic agent serve` gives an MCP client, such as Claude Code, tools to inspect your
datasets, find features, validate and plan a recipe, submit it, follow it and preview
its output. It needs the `mcp` extra:

```bash
pip install "mosaic-behavior[mcp]"
mosaic agent install --mcp
```

`--mcp` adds a `mosaic` server to the project's `.mcp.json`, serving the datasets in
and below the project directory, or the dataset that encloses it. Name others with
`--dataset PATH`, once per dataset. The other servers in `.mcp.json` are kept.

The server serves only the datasets it is given. Before it submits work that is costly,
covers the whole dataset or repeats an open request, it asks you to confirm. A dataset
that a mosaic service manages accepts no submission from it unless the server is
started with `--allow-managed-writes`. The server never overwrites, deletes or edits a
manifest. For those, the agent proposes the `mosaic` command for you to run.

`mosaic agent serve --config FILE` reads the datasets to serve from a YAML file, as
`datasets` (manifests or their directories), `discover` (directories searched for
datasets) and `allow_managed_writes`. The
[CLI reference](reference/cli.md#mosaic-agent-serve) lists every option.

## The guide in a terminal

```bash
mosaic agent guide
```

prints the guide, then the paths of this installation: the `mosaic` executable, the
Python interpreter, the local copy of these docs, and the dataset the current
directory is inside, if any.

## Agents inside the mosaic checkout

Work in a directory of your own, outside the mosaic clone. An agent started inside
the checkout reads the developer guide for changing mosaic itself,
[CLAUDE.md](https://github.com/EcodylicScience/mosaic/blob/main/CLAUDE.md), instead
of this setup.
