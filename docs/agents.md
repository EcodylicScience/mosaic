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
                       # and, with retire, the runs of retired recipes
cleanup:               # which superseded runs a cleanup keeps
  keep_newest: 1
  min_age_hours: 24
max_workers: 1         # the most entries a parallelizable step computes at once
allow_process_mode: false   # let a submission ask for process workers
min_free_memory_fraction: 0.10   # the share of host memory a parallel run keeps free
```

The tools name a dataset by the `uuid` in its manifest, which `list_datasets` reports.
The agent cannot open a dataset outside this list. A manifest without a `uuid`, or one
that does not read, is left out and reported on the server's standard error.

### The tools

| Tool | Result |
| --- | --- |
| `list_datasets`, `describe_dataset` | The served datasets, and one dataset's manifest fields, notes, tags, `splits` and counts |
| `list_sequences` | The entries, with the tracks table each resolves to |
| `get_inventory` | A dataset's computed artifacts, each with its status and coverage, and one artifact in depth with its params |
| `describe_step_types` | The features and ops a recipe may run, found by words, and the ones named with their params as a table |
| `validate_recipe`, `suggest_steps` | A recipe checked against the step types, and the steps or inputs that may be added |
| `patch_recipe` | A recipe changed by edits (add, remove or replace a step, set a param) and saved in the dataset, so the other tools take its digest instead of the document |
| `plan_recipe` | A recipe resolved against a dataset: each step's run id, status and coverage, the plan's cautions, and its fingerprint. `assume_added` names groups to treat as added, and the cautions then say which steps would fit again. With `cleanup: superseded`, the superseded runs of its storages and the saved recipes that `retire` can reclaim runs from. The parallelism a submission may ask for, with the host's processors and memory |
| `submit_pipeline` | Submits a planned recipe and starts it in the background. With `cleanup: superseded`, removes the superseded runs afterwards, and with `retire` the runs that retired recipes wrote. Takes `workers` and `parallel_mode` within your limits |
| `get_request_status`, `list_activity` | How far a request got, and the dataset's requests and attempts. Each step's workers and peak memory. `get_request_status` can wait for a step to end |
| `cancel_request` | Stops a request's running step and records the rest cancelled |
| `diagnose_failures` | Why a request or attempt failed, with the entries it lost |
| `preview_output` | One entry's table of a feature run, tracks variant or labels variant: its first rows, column statistics and file path, and a labels variant's events per class |

Only `submit_pipeline` and `cancel_request` change a dataset's data. `patch_recipe` saves
a recipe file under `.mosaic/pipelines/` and runs nothing. A submission must name the
fingerprint of a plan made against the dataset as it is now. The agent therefore plans
against the current dataset before it submits. Ask it to show you the plan. A submitted
request runs in a background process that outlives the server and the client. Follow it
with `mosaic pipeline status`, cancel it with `cancel_request` or
`mosaic cancel --request`, and resume a request that reads `orphaned` with
`mosaic pipeline run --manifest dataset.yaml --request <id>`.

An agent waits for a request with `get_request_status` and `wait_seconds` (45 at most).
The call returns as soon as a step of the request ends or the request does, and
otherwise when the time is up, so the agent needs no polling loop and no look at the
driver's process. With `brief` it lists only the steps that are running or did not
finish cleanly, and counts the others (`omitted_finished`, `omitted_pending`).
`mosaic pipeline status --request <id> --wait-seconds 45 --brief` does the same from a
shell. The driver log, `.mosaic/drivers/<id>.log` in the dataset, holds the output of
the driver and its steps. It explains a failure and does not report progress.

`get_request_status` also reports `poll_after_seconds`, how long to wait before reading
again without `wait_seconds`, and each step's phase and liveness. A request reads
`orphaned` when its driver stopped or its step's process is provably gone: the process
exited, its machine rebooted, or it wrote nothing for ten minutes. `plan_recipe` lists
the requests of the same recipe that are still running or orphaned. A step on another
machine whose liveness reads `unknown` was started by mosaic 0.17 or earlier and cannot
be checked.
Once you know its process has exited, record it with
`mosaic cancel --manifest dataset.yaml --request <id> --assume-exited`, then submit
the recipe again. The server offers no tool for this
([When a process is gone](guides/pipelines/run-and-see-what-ran.md#when-a-process-is-gone)).

`plan_recipe`, `submit_pipeline` and `get_inventory` take `entries` as a list of
`[group, sequence]` pairs or as the scope `--step-scope` takes: `{"entries": [...]}`, or
`{"groups": [...], "sequences": [...]}` with either left out. `plan_recipe` and
`submit_pipeline` also take `step_entries`, which narrows single steps by step id: a
classifier fitted on the training groups and scored on the test groups is one request
([Step scopes](guides/pipelines/chain-steps.md#narrow-one-step-step-scopes)). A step
scope is intersected with `entries`, so leave `entries` unset when steps are scoped to
different groups.

A long recipe is sent once. The agent writes a step repeated over seeds or classes once,
with `repeat` ([Repeat a step](guides/pipelines/chain-steps.md#repeat-a-step)), saves the
recipe with `patch_recipe`, and from then on passes `{"digest": "<digest>"}` (or
`{"saved": "<name>"}`) as the recipe and changes it with `patch_recipe` edits.
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
| `plan_recipe` | 3 missing entries per step. About 900 characters per step, so a plan of about 40 steps or more needs `limit` (and `cursor`) to be read a page of steps at a time |

A read whose result would be larger fails with reason `result_too_large`. Its detail
names the arguments that narrow the call (`narrow_with`), such as `columns` or
`max_columns` for a preview and `limit` for a plan. Every page of a plan carries the
whole plan's fingerprint, and its cautions are on the first page. `submit_pipeline` and `cancel_request` are never refused
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
  (`cleanup`). The question names how many runs, their size and their storages, and the
  retired recipes that wrote some of them. A hook that approves `cleanup` approves a
  retirement too;
- a step scores predictions against labels over a group that the dataset's
  [`splits`](dataset.md#hold-groups-out-of-training) seal (`score_sealed`). A sealed
  group allows one look, and this spends it. It fires whenever an `evaluate-labels` step
  is planned over entries of a sealed group, whether or not the group's labels are in the
  dataset. Keeping the labels outside the dataset until `mosaic pipeline score` adds them
  to a copy is the stronger protection.

`cancel_request` asks to confirm when a running step loses its work (`hard_cancel`): a
keypoint-MoSeq or VAME fit stops at once, and training loses the epoch in progress.

A plan over the whole dataset, or a step asked for more than 100 entries, does not
ask. `plan_recipe` reports each as a caution, and the agent shows you the cautions
with the plan before it submits. When the dataset gives its groups roles, the caution
for a whole-dataset plan names the groups it covers and their roles.

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

### Held-out groups

A dataset can mark groups as held out with [`splits`](dataset.md#hold-groups-out-of-training).
The agent reads them and never writes them: its tools cannot give a group a role or take
one away, so you set them with `mosaic splits`. With roles set:

- `plan_recipe` refuses a step that fits on the entries it covers when they include a
  `test` or `sealed` group, and `submit_pipeline` refuses the same recipe. The problem
  names the step, the groups and how to scope it. A step that applies a fitted model to
  a test group is not refused.
- `plan_recipe` takes `assume_added`, a list of groups, and notes in its cautions which
  steps would fit again and which would only re-score if those groups joined the
  dataset. It reads the recipe and changes nothing.
- `submit_pipeline` asks you to confirm a step that scores a sealed group
  (`score_sealed`).

To score data the agent has never seen, keep it outside the dataset and run
`mosaic pipeline score` yourself, on a copy of the dataset
([the protocol](guides/pipelines/chain-steps.md#score-on-held-out-data)).

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
- deleting tables, entries or runs, except superseded feature runs and the runs of
  recipes retired with them, which `submit_pipeline` removes with
  `cleanup: superseded` when the server runs with `--allow-cleanup feature-runs`
  (see below);
- `--apply` on `sweep-tracking`, `prune-media`, `prune-joined`, `reconcile` and
  `upgrade-tracks`;
- `reindex` and `release`;
- `mosaic cancel --assume-exited`, which records as cancelled an attempt that no
  machine can check, on your word that its process has exited;
- any change to the manifest: `init`, `sources`, `tags`, `notes`, `libraries`, `splits`;
- `mosaic pipeline score`, which adds held-out data to a copy of the dataset;
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
request of another recipe recorded it or pins it (unless it is a retired recipe that
wrote it, below), when an attempt that may still run
names it or may read it, when it covers entries or a fit its replacement does not, or
when a kept run or a step of the plan reads it. Recipes with the same non-empty
`name` are versions of each other, and an unnamed recipe is a version only of itself.
Name a recipe and keep the name across its versions, so a cleanup can remove the runs
of its earlier versions. Only the feature storages the recipe names are touched, unless
the call retires a recipe (below). A run of a named storage that was written outside a
request is a candidate whichever recipe it belongs to. Each removed run's `params.json`
is kept in the cleanup's run-log for computing the run again, and `get_request_status`
reports what was removed and kept. A managed dataset refuses a cleanup, and a queue
that runs the steps itself does not run one.

#### Retiring an earlier recipe

Rebuilding a recipe so that a step reads something else, or dropping a step, moves its
storage. The runs left in the old one are outside every later plan, and a recipe with
another name keeps the runs it recorded. To reclaim them, retire the earlier recipe:

1. `plan_recipe` with `cleanup: superseded` lists the saved recipes whose requests
   wrote runs under `retirable`: name, digest, request ids and how many runs.
2. Plan again with `retire: [<digest>]`. The runs that recipe's requests wrote are then
   candidates in any storage. Those in storages no step of the recipe names are listed
   under `cleanup.retired_runs`, and the rest on their steps, each with the retired
   digest.
3. Show the user the plan, then pass the same `retire` and the plan's fingerprint to
   `submit_pipeline`. The confirmation names the retired recipes.

`retirable` lists at most eight recipes, newest activity first, and counts the rest
under `retirable_omitted`; a recipe whose runs are all gone is not listed. When the
recipe you want is missing, take its digest from `list_activity`, whose requests carry
their `recipe_digest`. A call retires at most eight recipes, and `cleanup.retired_runs`
lists at most five runs, removable ones first, with the rest counted in
`cleanup.retired_omitted`.

A retirement applies to that one plan and submission, and the next plan judges as if
nothing was retired. It needs `cleanup: superseded`. It reaches only runs a retired
request wrote. A run that request only found already done, a run it pinned with `bind`,
and a run that a call outside every request, such as a notebook, also used keep their
protection. The other rules still apply to a retired run: a request's hold, a costly
feature, a reader, the policy's age, and `keep_newest` for each storage, so with the
default policy the newest retired run of each abandoned storage stays. An earlier
version of the planned recipe, one with the same name, never keeps a run, a retired one
included. The call is refused for a digest that no saved recipe has, for the recipe
being planned, and for a recipe with a request that has not ended.

### Parallel workers and memory

A feature step computes its entries one at a time. A feature that computes each entry on
its own, such as `speed-angvel` or `body-scale`, can compute several at once, and
`describe_step_types` reports it as `parallelizable`. A fit across every entry, such as
`global-kmeans`, and every op are not. The worker count belongs to the machine and not
to the recipe, so it changes neither a run's identity nor its outputs. A run that used
four workers is a cache hit for a run that uses one.

You set the limits in the configuration. The agent chooses within them and cannot raise
them:

- `max_workers` is the most entries a step computes at once, from 1 to 32. It is also
  the count a submission receives when it names none, and the effective count never
  exceeds the machine's processors. The default, 1, computes entries in turn.
- `allow_process_mode` lets a submission ask for `parallel_mode: process`. A submission
  that names no mode receives `thread`.
- `min_free_memory_fraction` is the share of the host's memory that a run with several
  workers keeps available. While the share is lower, the run waits for its entries in
  flight, whose tables free memory as they finish. When the share is still lower with
  none in flight, the run fails in a state that a later run resumes.

A submission over a limit is refused with the reason `parallel_limit`, and nothing is
recorded.

Threads share one copy of the feature and of the imported libraries, so memory grows by
about one entry's table per worker. A thread gains little for work that holds Python's
interpreter lock. Process workers use every core, and each is a separate interpreter that
imports the feature libraries and receives a pickled copy of each entry's table, so
memory grows by a whole interpreter and several copies of a table per worker. A
feature that does not fit in memory at four process workers may fit at two, or in
threads. A feature that defines `set_scope`, such as the interaction crop, runs in
threads whatever the mode is, because a process worker rebuilds the feature without it.

The agent sees what it needs to choose:

- `plan_recipe` reports the limits, the host's processors and memory, and for each step
  whether it is `parallelizable` and how many `workers` a default submission gives it.
- `get_request_status` reports each step's `workers`, `parallel_mode`, `peak_rss_bytes`
  and `min_available_bytes` while it runs and after. The peak is the resident memory of
  the step's process and its workers. A page that several processes map counts once for
  each, so process mode reads a little high.
- A worker that the host stops, usually for lack of memory, ends the step with
  `WorkerProcessDied`. The message gives the workers, the mode and the peak, and says
  to run again with fewer workers or in thread mode. Entries already written are kept,
  and the same recipe submitted again resumes.

The same options exist outside the server: `mosaic run --feature`, `mosaic run
--graph-request` and `mosaic pipeline run` take `--workers`, `--parallel-mode` and
`--min-free-memory-fraction`. A request that you resume by hand with
`mosaic pipeline run --request` runs its steps in turn unless you pass them. A queue
that dispatches the steps builds its own commands and does not pass them. When a queue
also runs several steps at once, the processors used are the steps times the workers.

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
