"""
AHFinder - Apparent Horizon Finder

Implementation of the apparent horizon location algorithm from
Huq, Choptuik & Matzner (2000) - arXiv:gr-qc/0002076

Also includes the Level Flow method from
Shoemaker, Huq & Matzner (2000) - arXiv:gr-qc/0004062

Uses Cartesian finite differences on a spherical surface parameterization
to avoid pole singularities.
"""

from .finder import ApparentHorizonFinder
from .surface import SurfaceMesh
from .levelflow import LevelFlowFinder

__version__ = "0.1.0"
__all__ = ["ApparentHorizonFinder", "SurfaceMesh", "LevelFlowFinder"]
