"""
Nonclassicality detection via linear programming.

Top-level package grouping the numerical solvers, analytical criteria, scan /
comparison drivers, visualization and shared utilities. See the subpackages:

- ``src.solvers``       : numerical nonclassicality witnesses (LP + minimize)
- ``src.analytical``    : analytical criteria and physical models
- ``src.comparison``    : numerical-vs-analytical parameter scans
- ``src.visualization`` : boundary extraction and plotting
- ``src.utils``         : shared numerical and I/O helpers
- ``src.config``        : tolerances, bounds, seeds and paths
"""

__version__ = "0.1.0"
