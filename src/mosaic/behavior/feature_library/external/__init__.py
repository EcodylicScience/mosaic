"""
External tool runners for mosaic.

The Python modules here are of two kinds. ``kpms_server.py`` and
``vame_runner.py`` are the runner programs. Each is launched by path as a
subprocess in a separate Python environment (keypoint-MoSeq's, built here, and
VAME's, built in ``vame-env/`` beside it). Mosaic does not import either, and
each is the only importer of its third-party library, **keypoint-MoSeq** or
**VAME**. The two libraries' licenses give different reasons for the
separation. keypoint-MoSeq is licensed by the Harvard University Office of
Technology Development for non-commercial research and academic use only;
mosaic is AGPL-3.0-or-later. AGPL section 7 forbids adding a restriction such
as "non-commercial only" to a covered work. A mosaic that imported
keypoint-moseq could not be distributed at all. Do not bundle keypoint-moseq
into a mosaic distribution or installer. VAME is licensed GPL-3.0, which does
not restrict use, and section 13 of the AGPL and of the GPL each permit
combining a work under the other. Mosaic's runner imports VAME in a process of
VAME's environment, and mosaic's distribution does not convey VAME. The GPL
source obligation falls on whoever conveys a built environment.

``runner_wire.py``, ``kpms_protocol.py`` and ``vame_protocol.py`` import only
the standard library and pydantic, crossing the boundary in both directions: a
runner imports them bare, as a script in the tool's environment, and mosaic
imports them as package modules (``runner_client.py``, ``kpms.py``,
``vame_feature.py``). A runner reads requests as JSON lines on its standard
input and writes replies the same way on its standard output, with an array
crossing as a file. See ``README.md`` here (for VAME, ``vame-env/README.md``)
for the setup, and ``NOTICE`` at the repository root for the terms.
"""
