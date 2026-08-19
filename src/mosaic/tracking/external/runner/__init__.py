"""The programs that run inside an Ultralytics environment, and their contract.

``ultralytics_protocol`` is imported from both sides of the boundary and takes no
import from ``ultralytics`` or from ``mosaic``. ``ultralytics_runner`` is a script
run by the external environment's interpreter, never imported by mosaic: it
resolves ``ultralytics_protocol`` as a bare top-level module from its own
directory, which only holds when it is the program being run.
"""
