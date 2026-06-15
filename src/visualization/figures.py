"""
Figure-building helpers shared across the presentation notebooks.

Phase 2 placeholder: the bulk of the plotting code currently lives inline in
``figures_lp.ipynb`` and ``mayorisation_functions.ipynb``. It is extracted into
reusable functions here during Phase 5 (notebook clean-up). For now this module
only re-exports the boundary plotting helper.
"""

from src.visualization.boundaries import extract_binary_boundary

__all__ = ["extract_binary_boundary"]
