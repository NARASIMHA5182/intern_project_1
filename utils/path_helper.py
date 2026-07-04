"""
utils/path_helper.py
--------------------
Utility for reliably resolving the project root directory and adding it to
``sys.path``. Import and call ``add_project_root()`` at the top of any
standalone script that lives inside a sub-package (e.g. preprocessing/,
training/, evaluation/) so that all intra-project imports resolve correctly
regardless of the current working directory.
"""

import os
import sys


def get_project_root() -> str:
    """Return the absolute path to the repository root.

    The root is defined as the directory two levels above this file:
        <root>/utils/path_helper.py  →  <root>

    Returns
    -------
    str
        Absolute path to the project root directory.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def add_project_root() -> str:
    """Append the project root to ``sys.path`` if it is not already present.

    Call this function once at the top of any script that needs to import
    project-level packages (e.g. ``utils``, ``preprocessing``, ``training``).

    Returns
    -------
    str
        The resolved project root path (useful for constructing file paths).
    """
    root = get_project_root()
    if root not in sys.path:
        sys.path.append(root)
    return root
