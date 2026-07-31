"""
External tool runners for mosaic.

Scripts in this directory bridge mosaic with packages that cannot live in the
mosaic environment. They run in a separate Python environment, invoked via
subprocess, and mosaic never imports them.

Today that package is **keypoint-MoSeq**, and the separation is a license
requirement rather than a packaging convenience. keypoint-MoSeq is licensed by
the Harvard University Office of Technology Development for non-commercial
research and academic use only; mosaic is AGPL-3.0-or-later. AGPL section 7
forbids adding a restriction such as "non-commercial only" to a covered work,
so a mosaic that imported keypoint-moseq could not be distributed at all.
Spawning a separate interpreter and exchanging JSON over a socket keeps them
two programs.

It follows that keypoint-moseq must never be bundled into a mosaic distribution
or installer. See ``README.md`` here for the setup, and ``docs/licensing.md``
for the terms.
"""
