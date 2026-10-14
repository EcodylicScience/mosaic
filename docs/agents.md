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
