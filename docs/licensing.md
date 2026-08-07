# Licensing

Mosaic is released under the **GNU Affero General Public License v3 or later**
(AGPL-3.0-or-later). See [LICENSE](https://github.com/EcodylicScience/mosaic/blob/main/LICENSE).

Mosaic also drives a number of third-party tools and models, and their terms are
not all the same as mosaic's. Two carry restrictions worth knowing before you
start: keypoint-MoSeq prohibits commercial use outright, and the strongest
available backbone for animal re-identification is non-commercial. This page
states which components carry restrictions that could affect whether you may use
them, and what mosaic does about it.

!!! note "Scope"

    This page covers the components mosaic installs through an extra, downloads
    at run time, or invokes as an external program, and it records what was read
    from each project's own license document. Ordinary permissive runtime
    dependencies (numpy, pandas, scikit-learn and the like) are not enumerated.
    It is a starting point for your own review, not legal advice, and not a
    completed audit of every transitive dependency.

## keypoint-MoSeq is non-commercial only

!!! warning "Commercial use is prohibited"

    keypoint-MoSeq is licensed by the Harvard University Office of Technology
    Development under a **Non-Commercial Research and Academic Use Software
    License**. Commercial use is expressly prohibited.

Harvard's definition of commercial use is broad. It reaches:

- use of the software in fee-for-service arrangements;
- use by core facilities or laboratories providing research services to, or in
  collaboration with, for-profit third parties for a fee;
- use in industry-sponsored or collaborative research projects where any
  commercial rights are granted to the sponsor or collaborator.

This is not a copyleft license, so there is no paid exception that cures it. It
restricts *who may run the software and for what purpose*. Read the terms
before use: <https://github.com/dattalab/keypoint-moseq/blob/main/LICENSE.md>

The license also asks users to acknowledge Harvard as the provider and to cite
the relevant publications, and to retain its copyright and other notices in the
software and in any derivative works.

### What mosaic does about it

Mosaic never installs, imports, bundles, or distributes keypoint-moseq. The
`kpms` feature runs it in a **separate Python environment you build yourself**,
reached over a Unix domain socket with a newline-delimited JSON protocol. No
licensed code is ever loaded into the mosaic process.

That separation is a license requirement, not an implementation convenience.
Mosaic is AGPL-3.0-or-later, and AGPL section 7 forbids adding a restriction
such as "non-commercial only" to a covered work. A mosaic that linked to or
imported keypoint-moseq would be a combined work that could not be distributed
at all. Keeping them two programs that exchange messages is what keeps mosaic
distributable. It follows that **keypoint-moseq must never be bundled into a
packaged mosaic installer**.

On top of that separation, `KpmsFeature` refuses to start the external process
until you confirm the terms apply to your use:

```bash
export MOSAIC_KPMS_LICENSE_ACCEPTED=1
```

Exactly `1` is accepted, and nothing else. Setting it is an assertion that your
use is permitted under Harvard's terms. Until then any attempt to run `kpms`
stops with the terms and a pointer here.

The setup instructions live in
[`src/mosaic/behavior/feature_library/external/README.md`](https://github.com/EcodylicScience/mosaic/blob/main/src/mosaic/behavior/feature_library/external/README.md).
They work from a source checkout only, because that directory's non-Python
files are not part of the installed wheel.

### Two clarifications worth recording

**A lock file is not distribution.** `external/uv.lock` records package names,
versions and hashes for a reproducible install. It is not the software, and
checking it in does not distribute keypoint-moseq. It should not be removed on
licensing grounds.

**Declaring the dependency does make mosaic an installer.** `external/pyproject.toml`
names keypoint-moseq, and the documented `uv sync` fetches it from PyPI under
Harvard's terms, with you as the licensee. That is exactly why the
acknowledgement is attached to the install step and to the first run, rather
than left implicit.

### Running mosaic as a service

A deployment that serves keypoint-MoSeq analyses to outside users for a fee is
the fee-for-service case Harvard's terms prohibit. `mosaic-api` runs mosaic
in-process and therefore inherits this refusal: its operator must set
`MOSAIC_KPMS_LICENSE_ACCEPTED` deliberately, on the machine, for the deployment
they are responsible for.

**Known gap.** The `GET /features` listing that `mosaic-api` serves, and the
feature picker in `mosaic-app` built on it, do not yet mark `kpms` as
restricted. A user browsing that interface sees it alongside unrestricted
features. Until that is fixed, this page and the refusal at run time are the
notice.

### If your use is not permitted

The `arhmm` feature fits a comparable autoregressive hidden Markov model in
mosaic's own code, with no keypoint-MoSeq or JAX dependency and no license
restriction. It is the intended alternative.

## Model weights carry their own license

**Mosaic distributes no model weights.** Every backbone is fetched at run time
from a source you name, and those weights carry their own license, independent
of mosaic's. AGPL-3.0-or-later covers this repository's source; it places no
restriction on the weights you load and confers no rights over them, and
conversely a non-commercial weights license does not become less restrictive by
being loaded from AGPL code.

`global-identity-embedding` makes that explicit: `model_name` takes any timm
architecture tag or Hugging Face hub id, and mosaic loads whatever it names.

| `model_name` | License | Pretraining | Commercial use |
| ------------ | ------- | ----------- | -------------- |
| `timm/swin_large_patch4_window12_384.ms_in22k_ft_in1k` (default) | MIT | ImageNet-22k, fine-tuned on ImageNet-1k | Permitted |
| `BVRA/MegaDescriptor-L-384` | CC-BY-NC-4.0 | 53 animal re-identification datasets | **Prohibited** |
| `BVRA/MegaDescriptor-T-224` | CC-BY-NC-4.0 | as above, much smaller and faster | **Prohibited** |

!!! warning "MegaDescriptor is non-commercial"

    `BVRA/MegaDescriptor-*` is released under
    [CC-BY-NC-4.0](https://huggingface.co/BVRA/MegaDescriptor-L-384), which does
    not permit commercial use. The restriction is on the *weights*, not on
    mosaic's code, and not on the architecture: MegaDescriptor is a stock
    `swin_large_patch4_window12_384` — the same Swin the default loads — trained
    on wildlife re-identification data. Only the trained parameters are
    restricted.

MegaDescriptor remains the right choice for academic wildlife
re-identification, where it substantially outperforms a generic ImageNet
backbone on telling individual animals apart, and it is one parameter away.
Selecting it, and complying with its terms, is your decision.

### Why this has no acceptance gate, when `kpms` does

The `kpms` refusal exists because merely running that feature *is* use of
restricted software, with no unrestricted path through it. Here the restricted
component is a value you type: the default is permissive, mosaic never selects
MegaDescriptor for you, and naming it is already the deliberate act an
environment variable would otherwise stand in for. A gate would add friction
without adding information.

**A note on accuracy, since it bears on the choice.** The default is the
permissively licensed option, not the most accurate one. MegaDescriptor's entire
value is that it was pretrained to tell individual animals apart; an ImageNet
backbone was pretrained to tell a cat from a bus. For near-identical laboratory
animals expect the default to be a weak zero-shot baseline and MegaDescriptor to
be a credible one. That ranking is the published result (Čermák et al., WACV
2024); nobody has measured either on mosaic's own data. Changing `model_name`
mints a new `run_id`, so the two runs coexist and you can compare the reported
top-1 from each.

## FERAL backbones

The `feral` extra installs FERAL itself under MIT, but the weights it runs on
are downloaded from the HuggingFace hub at first use and carry their own terms.
Which backbone `model_name` names decides which. Mosaic's default is
`vjepa2_vitl_diving48`.

| `feral.backbones` key | Resolves to | License |
| --------------------- | ----------- | ------- |
| `vjepa2_vitl_diving48` (mosaic default) | `facebook/vjepa2-vitl-fpc32-256-diving48` | MIT |
| `vjepa2_vitl_ssv2` | `facebook/vjepa2-vitl-fpc16-256-ssv2` | MIT |
| `vjepa2_vitl` | `facebook/vjepa2-vitl-fpc64-256` | MIT |
| `vjepa2_1_vitb_384`, `vjepa2_1_vitl_384`, `vjepa2_1_vitg_384`, `vjepa2_1_vitgg_384` | `facebookresearch/vjepa2` through `torch.hub` | MIT |
| `videoprism_v1_base` | `sposiboh/videoprism-base-f16r288-pt` | Apache-2.0 per the port's card — see below |
| `videoprism_v1_large` | `sposiboh/videoprism-large-f8r288-pt` | Apache-2.0 per the port's card — see below |

None of these is gated: there is no click-through and nothing to accept, so a
download is an ordinary anonymous fetch. The V-JEPA 2 repository states that
"The majority of V-JEPA 2 is licensed under MIT, however portions of the project
are available under separate license terms" — three source files are Apache-2.0.
That qualification is about the repository's code; every `facebook/vjepa2-*`
model card mosaic can reach declares MIT.

!!! warning "VideoPrism: an unreconciled provenance"

    FERAL's VideoPrism entries do not point at Google DeepMind. They point at
    the `sposiboh/videoprism-*-pt` repositories, third-party PyTorch ports
    whose cards describe themselves as a port of Google DeepMind's VideoPrism
    and declare `apache-2.0`.

    Upstream `google-deepmind/videoprism` licenses its **code** under Apache-2.0
    and states that "All other materials are licensed under the Creative Commons
    Attribution 4.0 International License (CC-BY)" — which is what covers the
    weights. The port re-declares Apache-2.0 over artifacts upstream places
    under CC-BY-4.0, and neither party has reconciled the two.

    Both licenses are permissive and CC-BY-4.0's substantive obligation is
    attribution, which this page discharges. If certainty matters for your use,
    take the weights from DeepMind directly and observe CC-BY-4.0.

## Third-party components

Read from each project's own license document. Verify against the version you
install before relying on any row.

| Component | Reached through | License | Commercial use |
| --------- | --------------- | ------- | -------------- |
| keypoint-MoSeq | environment you build under `feature_library/external`, run as a subprocess | Harvard OTD Non-Commercial Research and Academic Use | **Prohibited.** No paid exception exists |
| TRex | external binary in its own environment | AGPL-3.0-or-later (the `Application/src/commons` subtree under GPL-3.0) | **Company use requires a paid commercial license** from the authors |
| Ultralytics YOLO | `pose` extra, included in `recommended`; also what `mosaic track ultralytics` drives | AGPL-3.0 | Permitted under AGPL terms; an Enterprise license is sold for use that cannot meet them |
| `ultralytics-thop` | transitive dependency of Ultralytics; named in no `pyproject.toml` | AGPL-3.0-or-later | As Ultralytics |
| `lap` | `pose`, `polo` and `recommended` extras | BSD-2-Clause | Permitted |
| POLO | `polo` extra | AGPL-3.0 (fork of Ultralytics) | Permitted under AGPL terms. The Ultralytics Enterprise license covers Ultralytics' own distribution and does not extend to a third-party fork |
| SLEAP | external binary in its own environment | BSD 3-Clause Clear | Permitted |
| Lightning Pose | external binary in its own environment | MIT | Permitted |
| lightning-action | `lightning-action` extra | MIT | Permitted |
| FERAL | `feral` extra | MIT | Permitted |
| FERAL backbone weights | fetched at run time from the HuggingFace hub | per backbone; the default is MIT | See the backbone table below |
| DINOv2 (`dinov2_vits14`, `dinov2_vitb14`) | `identity` extra, fetched through `torch.hub` | Apache-2.0, code and weights | Permitted |
| timm | `identity` extra; the loader, not the weights | Apache-2.0 | Permitted |
| Backbone weights for `global-identity-embedding` | `identity` extra, fetched at run time from the hub id you name | whatever that repository states; the default is MIT | See the table above |
| FFmpeg / ffprobe | system binaries you install; invoked, never bundled | LGPL-2.1-or-later, or GPL if built with GPL-only components | Permitted. A redistributor who bundles an `ffmpeg` build must observe that build's terms |
| PyAV (`av`) | `mosaic-media[io]`, for in-process frame decoding | BSD-3-Clause | Permitted |
| `mosaic-media` | required dependency, same authors | Apache-2.0 | Permitted |

Two rows deserve a second look if you are working commercially.

**TRex** is dual-licensed: AGPL-3.0-or-later for open use, with company use
requiring a paid commercial license. Unlike keypoint-MoSeq, that restriction
*can* be cured by paying. Mosaic never bundles TRex — it invokes a binary you
installed — so it carries no gate here, but the obligation is yours.

**Ultralytics** ships in the curated `recommended` bundle under AGPL-3.0, and
Ultralytics sells an Enterprise license to users who cannot meet AGPL's terms.
Mosaic is AGPL-3.0-or-later itself, so there is no incompatibility; the question
is whether *your* use of the combined work can meet AGPL's obligations,
including its network-use provision.

## The tracker default tables

`mosaic track ultralytics` selects one of six Ultralytics tracker backends and
must name, in the run identifier, the exact settings the backend ran under. That
requires mosaic to hold its own copy of each backend's defaults: reading them
from the installed Ultralytics would make an upstream retune silently re-mint
every identifier already on disk, which is the failure the declared integration
version exists to prevent.

Mosaic therefore **transcribes** those defaults —
[`src/mosaic/tracking/ultralytics_track/tracker_defaults.py`](https://github.com/EcodylicScience/mosaic/blob/main/src/mosaic/tracking/ultralytics_track/tracker_defaults.py)
records the setting names and their values in mosaic's own typed structure, with
mosaic's own commentary. The Ultralytics YAML files themselves are not copied,
and no AGPL-licensed source is vendored into `src/`. The setting names are
dictated by the code that reads them, and the values are parameters rather than
expression. A test compares the transcription against whatever Ultralytics is
installed, so drift is a named failure at upgrade time.

## Citing the tools mosaic drives

Mosaic does not require citation, but several of the tools above ask for it, and
keypoint-MoSeq's license asks that Harvard be acknowledged as the provider and
the relevant publications cited. If you publish results produced with any of
these components, cite them as their authors ask.

TRex in particular asks to be cited as Walter, T. and Couzin, I. D. (2021),
"TRex, a fast multi-animal tracking system with markerless identification, and
2D estimation of posture and visual fields", *eLife* 10:e64000.

The six tracker backends `mosaic track ultralytics` selects between are each
someone's published method, and results produced with one should cite it:

- **ByteTrack** — Zhang, Y. *et al.* (2021), "ByteTrack: Multi-Object Tracking by
  Associating Every Detection Box", [arXiv:2110.06864](https://arxiv.org/abs/2110.06864).
- **BoT-SORT** — Aharon, N., Orfaig, R. and Bobrovsky, B.-Z. (2022), "BoT-SORT:
  Robust Associations Multi-Pedestrian Tracking",
  [arXiv:2206.14651](https://arxiv.org/abs/2206.14651).
- **OC-SORT** — Cao, J. *et al.* (2022), "Observation-Centric SORT: Rethinking
  SORT for Robust Multi-Object Tracking",
  [arXiv:2203.14360](https://arxiv.org/abs/2203.14360).
- **Deep OC-SORT** — Maggiolino, G. *et al.* (2023), "Deep OC-SORT:
  Multi-Pedestrian Tracking by Adaptive Re-Identification",
  [arXiv:2302.11813](https://arxiv.org/abs/2302.11813).
- **FastTracker** — Hashempoor, H. and Hwang, Y. D. (2025), "FastTracker:
  Real-Time and Accurate Visual Tracking",
  [arXiv:2508.14370](https://arxiv.org/abs/2508.14370).
- **TrackTrack** — Shim *et al.* (2025), "Focusing on Tracks for Online
  Multi-Object Tracking", *CVPR 2025*.

## Regenerating the dependency inventory

The tables above are curated: they name the components that carry an obligation.
The resolved dependency set behind them is derived rather than curated, and can
be regenerated at any time:

```bash
python scripts/gen_third_party_inventory.py            # flagged rows only
python scripts/gen_third_party_inventory.py --all      # every distribution
python scripts/gen_third_party_inventory.py --offline  # no network
```

It reads `uv.lock` and the isolated keypoint-MoSeq lock, walks the transitive
closure per extra and per dependency group, resolves each license from PyPI
metadata, and flags whatever is not permissive. It writes nothing and hard-codes
no counts — the counts come out of the lock, so they cannot go stale silently.

Two things it deliberately does not do. It never asks PyPI about a package
resolved from git or a path, because the name there can belong to different
code: `ultralytics` in this lock is the POLO fork, and PyPI serves a real
release under that same name and version. And it reports "proprietary" and
"non-commercial" as separate verdicts, because the CUDA runtime libraries that
arrive with GPU PyTorch are the former and not the latter.

## Attributions

The obligations above, and the citations owed, are recorded in
[NOTICE](https://github.com/EcodylicScience/mosaic/blob/main/NOTICE) at the
repository root. AGPL v3 section 5 requires that file to be preserved in
modified and redistributed versions.
