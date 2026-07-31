# Licensing

Mosaic is released under the **GNU Affero General Public License v3 or later**
(AGPL-3.0-or-later). See [LICENSE](https://github.com/EcodylicScience/mosaic/blob/main/LICENSE).

Mosaic also drives a number of third-party tools and models, and their terms are
not all the same as mosaic's. One of them — keypoint-MoSeq — prohibits
commercial use outright. This page states which components carry restrictions
that could affect whether you may use them, and what mosaic does about it.

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

## Third-party components

Read from each project's own license document. Verify against the version you
install before relying on any row.

| Component | Reached through | License | Commercial use |
| --------- | --------------- | ------- | -------------- |
| keypoint-MoSeq | environment you build under `feature_library/external`, run as a subprocess | Harvard OTD Non-Commercial Research and Academic Use | **Prohibited.** No paid exception exists |
| TRex | external binary in its own environment | AGPL-3.0-or-later (the `Application/src/commons` subtree under GPL-3.0) | **Company use requires a paid commercial license** from the authors |
| Ultralytics YOLO | `pose` extra, included in `recommended` | AGPL-3.0 | Permitted under AGPL terms; an Enterprise license is sold for use that cannot meet them |
| POLO | `polo` extra | AGPL-3.0 (fork of Ultralytics) | Permitted under AGPL terms |
| SLEAP | external binary in its own environment | BSD 3-Clause Clear | Permitted |
| Lightning Pose | external binary in its own environment | MIT | Permitted |
| lightning-action | `lightning-action` extra | MIT | Permitted |
| FERAL | `feral` extra; V-JEPA 2 backbone | MIT (some V-JEPA 2 files Apache-2.0) | Permitted |
| DINOv2 (`dinov2_vits14`, `dinov2_vitb14`) | `identity` extra, fetched through `torch.hub` | Apache-2.0, code and weights | Permitted |

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

## Citing the tools mosaic drives

Mosaic does not require citation, but several of the tools above ask for it, and
keypoint-MoSeq's license asks that Harvard be acknowledged as the provider and
the relevant publications cited. If you publish results produced with any of
these components, cite them as their authors ask.
