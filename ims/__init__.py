"""
Module for chemometrics with GC-IMS data
======================================

Documentation is available via docstrings in classes and functions.

Provides:
---------
* Fast IO including a mea file reader
(standard file format for G.A.S Dortmund instruments).

* Preprocessing steps and utilities (alignment, resampling, plotting etc.).

* Scripted statistical workflows with prebuilt plots
for common algorithms.
"""
__version__ = "0.1.4"
__author__ = "Competency Center for Chemometrics Mannheim"
__credits__ = "Competency Center for Chemometrics Mannheim"

from ims.gcims import Spectrum
from ims.dataset import Dataset
from ims.pca import PCA_Model
from ims.plsr import PLSR
from ims.plsda import PLS_DA
from ims.hca import HCA
import ims.utils
