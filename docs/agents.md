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

`mosaic agent serve` is an MCP server that gives an MCP client, such as Claude Code or
Claude Desktop, tools for your mosaic datasets. The agent can describe a dataset, find
and describe features, draft, validate and plan a recipe, submit it, follow it, cancel
it, diagnose its failures and preview its output. The server runs on your machine,
calls the mosaic library directly, and serves one client over standard input and
output.

### Installation

Install the `mcp` extra into the conda environment where mosaic is installed. `all`
does not include it.

```bash
conda activate mosaic
pip install "mosaic-behavior[mcp]"     # in a checkout: pip install -e ".[all,mcp]"
```

The [one-ffmpeg rule](installation.md#one-ffmpeg-build-per-environment) applies to any
process that decodes media. The server reads datasets without loading OpenCV or PyAV,
but the steps it submits run in the same environment and decode video. Use the
environment built by the [installation](installation.md) page.

### Registration in Claude Code

```bash
mosaic agent install --mcp
```

`--mcp` adds a `mosaic` server to the project's `.mcp.json`, serving the datasets in
and below the project directory, or the dataset that encloses it. Name others with
`--dataset PATH`, once per dataset. The entry names your `mosaic` executable by its full
path, and the other servers in `.mcp.json` are kept. Run the command again after you
move or rebuild the environment. To remove the server, delete the `mosaic` entry of
`.mcp.json`.

Submitted steps run with the server's environment, and Claude Code passes the
environment of its shell to the server. Start Claude Code in a shell where the mosaic
environment is active, with `ffmpeg` on `PATH` and the variables of the
[tools mosaic runs](installation.md#tools-that-run-in-their-own-environment) exported.
A step started without them fails in the background. An `env` object in the `mosaic`
entry of `.mcp.json` sets them too, but `mosaic agent install --mcp` replaces the
entry and drops it.

### Registration in Claude Desktop

Claude Desktop reads its servers from `claude_desktop_config.json`: on macOS in
`~/Library/Application Support/Claude/`, on Windows in `%APPDATA%\Claude\`. Settings,
Developer, Edit Config opens it. Add a `mosaic` entry. Its `command` is the full path of
your `mosaic` executable, as `which mosaic` prints it in the activated environment:

```json
{
  "mcpServers": {
    "mosaic": {
      "command": "/Users/you/miniforge3/envs/mosaic/bin/mosaic",
      "args": ["agent", "serve", "--dataset", "/Users/you/data/study1"],
      "env": {
        "PATH": "/Users/you/miniforge3/envs/mosaic/bin:/Users/you/miniforge3/condabin:/usr/bin:/bin",
        "MOSAIC_TREX_CONDA_ENV": "trex"
      }
    }
  }
}
```

Claude Desktop starts the server without your shell's environment, where a bare `mosaic`
does not resolve. Submitted steps run with the server's environment. `env` therefore
puts the environment's `bin` directory, with `ffmpeg` and `ffprobe` in it, on `PATH`. A
tool named by `MOSAIC_<TOOL>_CONDA_ENV` also needs `conda` on that `PATH`, from the
`condabin` directory of your conda installation. A tool named by its absolute
`MOSAIC_<TOOL>_BIN` does not. `env` also sets the variables of the
[tools mosaic runs](installation.md#tools-that-run-in-their-own-environment) in use, and
`MOSAIC_KPMS_LICENSE_ACCEPTED` once you accept the keypoint-MoSeq license. A step
started without them fails in the background. Restart Claude Desktop after editing the
file.

### Datasets the server serves

The server serves only the datasets it is started with. Each `--dataset PATH` names
one dataset, by its manifest or its directory. `--config FILE` reads a YAML file
instead, whose relative paths are read from the file's directory:

```yaml
datasets:              # manifests, or the directories that contain them
  - ~/data/study1
discover:              # directories searched for datasets, three levels deep
  - ~/data/projects
allow_managed_writes: false
```

The tools name a dataset by the `uuid` in its manifest, which `list_datasets` reports.
The agent cannot open a dataset outside this list. A manifest without a `uuid`, or one
that does not read, is left out and reported on the server's standard error.

### The tools

| Tool | Result |
| --- | --- |
| `list_datasets`, `describe_dataset` | The served datasets, and one dataset's manifest fields, notes, tags and counts |
| `list_sequences` | The entries, with the tracks table each resolves to |
| `get_inventory` | A dataset's computed artifacts, each with its status, coverage and params |
| `search_step_types`, `describe_step_type` | The features and ops a recipe may run, and one in full with its params |
| `validate_recipe`, `suggest_steps` | A recipe checked against the step types, and the steps or inputs that may be added |
| `plan_recipe` | A recipe resolved against a dataset: each step's run id, status and coverage, and the plan's fingerprint |
| `submit_pipeline` | Submits a planned recipe and starts it in the background |
| `get_request_status`, `list_activity` | How far a request got, and the dataset's requests and attempts |
| `cancel_request` | Stops a request's running step and records the rest cancelled |
| `diagnose_failures` | Why a request or attempt failed, with the entries it lost |
| `preview_output` | One entry's table of a feature run or tracks variant: its first rows, column statistics and file path |

Only `submit_pipeline` and `cancel_request` change a dataset. A submission must name the
fingerprint of a plan made against the dataset as it is now. The agent therefore plans
against the current dataset before it submits. Ask it to show you the plan. A submitted
request runs in a background process that outlives the server and the client. Follow it
with `mosaic pipeline status`, cancel it with `cancel_request` or
`mosaic cancel --request`, and resume a request that reads `orphaned` with
`mosaic pipeline run --manifest dataset.yaml --request <id>`.

The server also offers resources (the agent guide, the step-type catalog, and per
dataset its manifest, inventory, recipes, requests, run-logs and recorded params) and
prompts for building a pipeline, diagnosing a dataset, explaining a request, comparing
runs and summarizing a dataset.

The server reads which features and ops this installation can run once, when it starts.
Restart it after installing an extra or building a tool's environment.

### Confirmation before costly work

`submit_pipeline` asks you to confirm a submission when:

- a step runs on a GPU lane, trains a model, or runs a tracker or model inference
  (TRex requires a paid license for use by a company);
- `allow_partial` lets a step run over fewer entries than planned, or `bind` pins
  artifacts that the recipe did not produce;
- a step is asked for more than 100 entries, or the submission covers the whole
  dataset;
- a request of the same recipe over overlapping entries has not ended.

`cancel_request` asks to confirm when a running step loses its work: a keypoint-MoSeq
or VAME fit stops at once, and training loses the epoch in progress.

A client that supports MCP elicitation shows you a form listing the conditions, and
only your answer confirms. With any other client, the tool refuses, the agent asks you
in the conversation, and it repeats the call with `confirm` set once you agree.

A dataset whose manifest names a managing service (`managed_by`), such as a mosaic-api
deployment, refuses both tools unless the server is started with
`--allow-managed-writes` or `allow_managed_writes: true`. Every read tool still works.

### Operations the server leaves to you

These change or remove what other work depends on, or accept terms on your behalf.
The server does not expose them as tools. When one is needed, the agent proposes the
`mosaic` command or the library call, and you run it:

- overwriting a computed run (`overwrite`);
- deleting tables, entries or runs;
- `--apply` on `sweep-tracking`, `prune-media`, `prune-joined`, `reconcile` and
  `upgrade-tracks`;
- `reindex` and `release`;
- any change to the manifest: `init`, `sources`, `tags`, `notes`, `libraries`;
- `extract-frames`, whose frames annotation tools refer to by path;
- accepting the keypoint-MoSeq license (`MOSAIC_KPMS_LICENSE_ACCEPTED`).

### Data sent to the model provider

Every tool result, and every resource the client reads, goes to your client's model.
With a hosted model, that is the model's provider. The resources include the manifest
with its notes and tags, run-logs, recorded params and request files. `preview_output`
returns up to 50 rows of a table and statistics of its columns, `describe_dataset`
returns the manifest's notes and tags, and the other tools return entry names, params,
paths and error messages. Serve only datasets whose values may leave your machine, or
use a client with a local model.

### Shared deployments

The server is built for one user on a personal machine. On a shared JupyterHub, it runs
work outside the mosaic service's accounts, permissions and run records, and a culled
server kills the background process that runs a submitted request. Such deployments
should use the server that the mosaic service hosts, once it is available.

### Every option

`mosaic agent serve --help` and the [CLI reference](reference/cli.md#mosaic-agent-serve)
list every option.

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
