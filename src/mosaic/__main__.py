"""``python -m mosaic``: the ``mosaic`` CLI, reached through the interpreter.

An agent or a script whose shell has no environment activated can run
``<python> -m mosaic`` where the ``mosaic`` script is not on ``$PATH``.
"""

from __future__ import annotations

from mosaic.cli import app

if __name__ == "__main__":
    app(prog_name="mosaic")
