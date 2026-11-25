# Try mosaic with Claude

Install mosaic, download a small example dataset and ask Claude about it, on your own
computer. This page uses Claude Desktop. Claude Code users follow the same steps with
the registration command of [step 3](#3-register-the-server).

## 1. Install

Build the conda environment from the [installation page](installation.md). mosaic
decodes video through one ffmpeg build per environment, and pip alone cannot arrange
that, so conda installs the media libraries first. Then add the MCP server inside that
environment:

```bash
conda activate mosaic
pip install "mosaic-behavior[mcp]"
```

## 2. Get a dataset

```bash
mosaic examples list
mosaic examples fetch calms21
```

`list` prints each example with its download size, license and the notebook that uses
it, and reads nothing from the network. `fetch` downloads one, verifies it, unpacks it
and prints the path of its manifest. Running it again reuses a complete download.

| Example   | Download | What it holds                                                         |
| --------- | -------- | --------------------------------------------------------------------- |
| `calms21` | 12.6 MB  | CalMS21 mouse pairs: raw tracks and behavior labels, still to convert |
| `shiners` | 358.8 MB | 10 shiner school recordings with converted tracks, and a raw sample   |

An example unpacks into `~/mosaic-examples/<name>/`. Name another directory with
`--dir`. The dataset of `calms21` is `~/mosaic-examples/calms21/calms21`, and that of
`shiners` is `~/mosaic-examples/shiners/shiners-tracks`.

## 3. Register the server

### Claude Desktop

Quit Claude Desktop, so that it does not write its configuration over the entry. Then
register the server with the dataset:

```bash
mosaic agent install --desktop --trust-pickled-files \
    --dataset ~/mosaic-examples/calms21/calms21
```

For `shiners`, leave out `--trust-pickled-files`:

```bash
mosaic agent install --desktop \
    --dataset ~/mosaic-examples/shiners/shiners-tracks
```

The command writes the `mosaic` entry of Claude Desktop's configuration file, keeps the
file's other servers and prints what it wrote. Start Claude Desktop, and check Settings,
Developer for a running `mosaic` server. Run the command again after you move or rebuild
the environment. A second run reports `unchanged` when the entry survived.

CalMS21's raw `.npy` files load only by unpickling, which can run code in the file.
`--trust-pickled-files` lets the server unpickle the raw files of every dataset it
serves, so give it only datasets from sources you trust. The `shiners` tracks are
already converted and need no flag.

The command writes the file in the usual place of your system: on macOS
`~/Library/Application Support/Claude/`, on Linux `~/.config/Claude/`, and on Windows
`%APPDATA%\Claude\`, or where the Microsoft Store build keeps it. It refuses when Claude
Desktop's directory does not exist, which means Claude Desktop is not installed or has
never been opened, and on Windows when it finds several candidate files. Name the file
with `--desktop-config` in either case, or when the app reads another one:

```bash
mosaic agent install --desktop --desktop-config FILE --dataset DATASET
```

On Windows, check Settings, Developer after the restart. When the `mosaic` server is
missing, the app reads another file: run the command again with `--desktop-config`.

### Claude Code

In a project directory:

```bash
mosaic agent install --mcp --dataset ~/mosaic-examples/calms21/calms21
```

Claude Code passes its shell's environment to the server. For CalMS21, export the trust
setting in the shell that starts Claude Code, before it starts:

```bash
export MOSAIC_TRUST_PICKLED_FILES=1
```

## 4. Your own dataset

Skip the download and name your dataset's manifest, or its directory, with `--dataset`.
That covers a dataset downloaded from the mosaic app or API, and one built with
`mosaic init` or in Python. Repeat the option to serve several datasets.

## 5. First prompts

For CalMS21, after registering it with `--trust-pickled-files`:

> Describe the calms21 dataset and what it still needs.

> Convert its tracks and labels, then show me the plan before running it.

For shiners:

> Compute the polarization of each group size and compare them.

Claude asks you before it runs anything that takes long or changes the dataset.

## 6. What stays on your computer

The server runs on your machine and reads your files there. What the tools return goes
to the model your client uses: see
[Data sent to the model provider](agents.md#data-sent-to-the-model-provider). Work that
costs time, writes to a dataset or deletes runs waits for your answer, as
[Confirmation before costly work](agents.md#confirmation-before-costly-work) describes.
[Work with an AI coding agent](agents.md) covers the server's other tools and options.
