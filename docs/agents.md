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
pip install "mosaic-behavior[mcp]"
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
allow_cleanup: []      # [feature-runs] lets submit_pipeline remove superseded runs
cleanup:               # which superseded runs a cleanup keeps
  keep_newest: 1
  min_age_hours: 24
```

The tools name a dataset by the `uuid` in its manifest, which `list_datasets` reports.
The agent cannot open a dataset outside this list. A manifest without a `uuid`, or one
that does not read, is left out and reported on the server's standard error.

### The tools

| Tool | Result |
| --- | --- |
| `list_datasets`, `describe_dataset` | The served datasets, and one dataset's manifest fields, notes, tags and counts |
| `list_sequences` | The entries, with the tracks table each resolves to |
| `get_inventory` | A dataset's computed artifacts, each with its status and coverage, and one artifact in depth with its params |
| `describe_step_types` | The features and ops a recipe may run, found by words, and the ones named with their params as a table |
| `validate_recipe`, `suggest_steps` | A recipe checked against the step types, and the steps or inputs that may be added |
| `plan_recipe` | A recipe resolved against a dataset: each step's run id, status and coverage, the plan's cautions, and its fingerprint. With `cleanup: superseded`, the superseded runs of its storages |
| `submit_pipeline` | Submits a planned recipe and starts it in the background. With `cleanup: superseded`, removes the superseded runs afterwards |
| `get_request_status`, `list_activity` | How far a request got, and the dataset's requests and attempts |
| `cancel_request` | Stops a request's running step and records the rest cancelled |
| `diagnose_failures` | Why a request or attempt failed, with the entries it lost |
| `preview_output` | One entry's table of a feature run, tracks variant or labels variant: its first rows, column statistics and file path, and a labels variant's events per class |

Only `submit_pipeline` and `cancel_request` change a dataset. A submission must name the
fingerprint of a plan made against the dataset as it is now. The agent therefore plans
against the current dataset before it submits. Ask it to show you the plan. A submitted
request runs in a background process that outlives the server and the client. Follow it
with `mosaic pipeline status`, cancel it with `cancel_request` or
`mosaic cancel --request`, and resume a request that reads `orphaned` with
`mosaic pipeline run --manifest dataset.yaml --request <id>`.

`get_request_status` reports `poll_after_seconds`, how long the agent waits before it
polls again, and each step's phase and liveness. A request reads `orphaned` when its
driver stopped or its step's process is provably gone: the process exited, its machine
rebooted, or it wrote nothing for ten minutes. `plan_recipe` lists the requests of the
same recipe that are still running or orphaned. A step on another machine whose
liveness reads `unknown` was started by mosaic 0.17 or earlier and cannot be checked.
Once you know its process has exited, record it with
`mosaic cancel --manifest dataset.yaml --request <id> --assume-exited`, then submit
the recipe again. The server offers no tool for this
([When a process is gone](guides/pipelines/run-and-see-what-ran.md#when-a-process-is-gone)).

`plan_recipe`, `submit_pipeline` and `get_inventory` take `entries` as a list of
`[group, sequence]` pairs or as `{"groups": [...]}`. `plan_recipe` and
`submit_pipeline` also take `step_entries`, which narrows single steps by step id: a
classifier fitted on the training groups and scored on the test groups is one request
([Step scopes](guides/pipelines/chain-steps.md#narrow-one-step-step-scopes)).
Conversion is a recipe step too: `convert-tracks` and `convert-labels` convert the raw
files a scan recorded. Declaring sources and scanning stay with you, so the agent
proposes `mosaic sources add` and `mosaic scan` when a dataset has nothing scanned.

The server also offers resources (the agent guide, the step-type catalog, and per
dataset its manifest, inventory, recipes, requests, run-logs, recorded params and a
run's report, such as the `metrics.json` that `evaluate-labels` writes) and prompts
for building a pipeline, diagnosing a dataset, explaining a request, comparing runs
and summarizing a dataset. A report is read, not computed, so it takes no tool. The
guide resource is the same text as `mosaic agent guide` and the installed `SKILL.md`,
so an agent reads it once.

### Result sizes

Each result goes into the model's context, so the read tools return what an agent
needs to choose its next call, and the server refuses a result above a budget.

| Result | Bound |
| --- | --- |
| Any read tool | 40,000 characters |
| `describe_step_types` with `names` | Up to 8 step types, each a params table of at most about 16,000 characters |
| `preview_output` | 5 rows and 40 columns by default (`limit`, `max_columns`), the identity and bookkeeping columns first |
| `get_inventory` listing | No params, and 5 missing entries per artifact. A `ref` returns one artifact with its params |
| `plan_recipe` | 3 missing entries per step |

A read whose result would be larger fails with reason `result_too_large`. Its detail
names the arguments that narrow the call (`narrow_with`), such as `columns` or
`max_columns` for a preview. `submit_pipeline` and `cancel_request` are never refused
for their size, because their work is done by the time they return.

A preview's `columns` takes names or `fnmatch` patterns (`W_*_f0`), and the columns
left unread are named in `columns_omitted`. `describe_step_types` returns the params
JSON Schema only under `detail: "schema"`.

A successful result carries the output twice: as structured content, and as JSON text
after the summary, for a client that reads only text. `describe_step_types`,
`plan_recipe`, `submit_pipeline`, `get_request_status` and `cancel_request` are
marked `anthropic/alwaysLoad`, so Claude Code loads them without a tool search. The
server sets no `anthropic/maxResultSizeChars`, because it bounds its results itself.

The server reads which features and ops this installation can run once, when it starts.
Restart it after installing an extra or building a tool's environment.

### Confirmation before costly work

`submit_pipeline` asks you to confirm a submission when:

- a step runs on a GPU lane (`gpu_lane`), trains a model (`training_op`), or runs a
  tracker or model inference (`tracking_op`; TRex requires a paid license for use by
  a company);
- `allow_partial` lets a step run over fewer entries than planned (`allow_partial`),
  or `bind` pins artifacts that the recipe did not produce (`bind`);
- a request of the same recipe over overlapping entries has not ended
  (`open_duplicate`);
- the submission removes superseded runs of its steps' storage after its steps finish
  (`cleanup`). The question names how many runs, their size and their storages.

`cancel_request` asks to confirm when a running step loses its work (`hard_cancel`): a
keypoint-MoSeq or VAME fit stops at once, and training loses the epoch in progress.

A plan over the whole dataset, or a step asked for more than 100 entries, does not
ask. `plan_recipe` reports each as a caution, and the agent shows you the cautions
with the plan before it submits.

A client that supports MCP elicitation shows you a form listing the conditions, and
only your answer confirms. With any other client, the tool refuses with reason
`confirmation_required`, the agent asks you in the conversation, and it repeats the
call with `confirm` set once you agree. Your answer confirms only the conditions the
form named. A call that meets another condition by the time it runs, such as a request
of the same recipe started since, shows the form again. The form has two other
outcomes:

- `confirmation_declined`: the form was declined, or accepted with its box unticked.
- `confirmation_unanswered`: the client cancelled the form, as it does when nobody can
  see it, such as in `claude -p`.

The call did not run in either case. The agent stops and reports. It does not repeat
the call, or run the same work through the `mosaic` CLI, unless you ask.

The last line of the form's message names the tool and the conditions, sorted, for a
hook to read:

```text
[mosaic-confirm tool=submit_pipeline conditions=gpu_lane,training_op]
```

A dataset whose manifest names a managing service (`managed_by`), such as a mosaic-api
deployment, refuses both tools unless the server is started with
`--allow-managed-writes` or `allow_managed_writes: true`. Every read tool still works.

### Unattended runs: approving conditions with a hook

An agent run without a person, such as `claude -p`, cannot answer a form. An operator
approves conditions in advance with a Claude Code `Elicitation` hook that runs
`mosaic agent confirm-hook`. The command accepts a confirmation whose conditions are
all approved. It answers any other as `--otherwise` says: `decline` (the default),
`cancel`, or `ask`, which leaves the form to the dialog. A confirmation whose last
line it cannot read, such as one from a mosaic too old to write that line, is answered
the same way.

```json
{
  "hooks": {
    "Elicitation": [
      {
        "matcher": "mosaic",
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/env/bin/mosaic agent confirm-hook --approve gpu_lane --approve training_op --otherwise decline"
          }
        ]
      }
    ]
  }
}
```

The matcher is the server's name in `.mcp.json`. Pass `--server` when that name is not
`mosaic`. `--approve` takes the condition names above, once per condition. An unknown
name fails when you first run the command.

An agent that can edit its own settings can add a hook that accepts everything. A
decline from any hook overrides an accept from another, so `--otherwise decline` keeps
such a hook from approving a condition you did not list. `cancel` and `ask` give no
such guarantee. Keep the operator's file out of the agent's reach:

- Save it outside the directories the agent can write, mounted read-only in a
  container, and pass it with `claude --settings /path/to/operator-settings.json`.
- Or install it as managed settings.

The hook answers only the server it is named for, so keep the server's configuration
out of reach too. Otherwise the agent can list mosaic under another name, which your
hook does not match. Pass the configuration from the same read-only place with
`claude --mcp-config /path/to/operator-mcp.json --strict-mcp-config`.

`--bare` skips hooks, so do not use it for such a run. `claude --debug hooks` logs each
elicitation that a hook resolved.

### Operations the server leaves to you

These change or remove what other work depends on, or accept terms on your behalf.
The server does not expose them as tools. When one is needed, the agent proposes the
`mosaic` command or the library call, and you run it:

- overwriting a computed run (`overwrite`);
- deleting tables, entries or runs, except superseded feature runs, which
  `submit_pipeline` removes with `cleanup: superseded` when the server runs with
  `--allow-cleanup feature-runs` (see below);
- `--apply` on `sweep-tracking`, `prune-media`, `prune-joined`, `reconcile` and
  `upgrade-tracks`;
- `reindex` and `release`;
- `mosaic cancel --assume-exited`, which records as cancelled an attempt that no
  machine can check, on your word that its process has exited;
- any change to the manifest: `init`, `sources`, `tags`, `notes`, `libraries`;
- `extract-frames`, whose frames annotation tools refer to by path;
- accepting the keypoint-MoSeq license (`MOSAIC_KPMS_LICENSE_ACCEPTED`).

### Reclaiming superseded runs

A recipe whose params changed leaves its earlier runs in the same storages. The agent
can remove them only on a server started with `--allow-cleanup feature-runs`, or with
`allow_cleanup: [feature-runs]` in its configuration. `plan_recipe` with
`cleanup: superseded` lists them on any server, because it only reads:

- each step lists up to five other runs of its storage under `superseded`, removable
  ones first, each removable or kept with its reason, and `cleanup` totals them all;
- `submit_pipeline` with `cleanup: superseded` and that plan's fingerprint asks you to
  confirm (`cleanup`), then removes the removable runs once every step has finished.

The policy is yours, in the configuration's `cleanup`, and no tool call changes it. The
defaults keep the newest superseded run of each storage and every run younger than
24 hours. A sandbox that reclaims space within one session might use:

```yaml
allow_cleanup: [feature-runs]
cleanup:
  keep_newest: 0
  min_age_hours: 1
```

A run is always kept when its feature is costly to fit again, when its `params.json`
is missing, when a running or recent request holds it (for at least an hour), when a
request of another recipe recorded it or pins it, when an attempt that may still run
names it or may read it, when it covers entries or a fit its replacement does not, or
when a kept run or a step of the plan reads it. Two recipes are versions of each other
when they have the same digest or the same non-empty `name`. Name a recipe and keep
the name across its versions. A cleanup then removes the runs of its earlier versions.
Only the feature storages the recipe names are touched. A run in a storage the recipe
no longer names is not removed, and a run of a named storage that was written outside
a request is a candidate whichever recipe it belongs to. Each removed run's
`params.json` is kept in the cleanup's run-log for computing the run again, and
`get_request_status` reports what was removed and kept. A managed dataset refuses
a cleanup, and a queue that runs the steps itself does not run one.

### Data sent to the model provider

Every tool result, and every resource the client reads, goes to your client's model.
With a hosted model, that is the model's provider. The resources include the manifest
with its notes and tags, run-logs, recorded params and request files. `preview_output`
returns up to 50 rows of a table and statistics of its columns, and of a labels
variant its events and class names, `describe_dataset`
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
