# Label Library

Label format converters for importing behavior annotations from external tools
into mosaic's standardized NPZ format.

Supported formats include CalMS21, BORIS aggregated CSV/TSV, and BORIS
Pandas pickle. `register_label_converter` is **called** on each converter
class in `label_library/__init__.py`, not used as a decorator, so the
registry holds exactly what that file names.

::: mosaic.behavior.label_library
    options:
      show_source: true
