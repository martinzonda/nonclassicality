"""
Boundary extraction and plotting helpers for 2D nonclassicality maps.

``extract_binary_boundary`` is extracted verbatim (Phase 2) from
``optimization_reworked.ipynb``. It uses a matplotlib contour at level 0.5 to
recover the ordered boundary curves separating the classical (0) and
nonclassical (1) regions of a binary map.

NOTE: naming, type hints, docstrings and logging are handled in Phase 3.
"""

import matplotlib.pyplot as plt


def extract_binary_boundary(P0_all, P1_all, nonclass_num):
    """
    Returns ordered boundary curves between 0 and 1 regions.

    Output:
        boundaries = [array([[P0, P1], ...]), ...]
    """
    fig, ax = plt.subplots()

    cs = ax.contour(P0_all, P1_all, nonclass_num, levels=[0.5])

    boundaries = []
    for path in cs.get_paths():
        v = path.vertices
        if len(v) > 1:
            boundaries.append(v.copy())

    plt.close(fig)
    return boundaries
