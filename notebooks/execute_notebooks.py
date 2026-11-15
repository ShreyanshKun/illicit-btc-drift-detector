"""Execute all notebooks in place so that outputs are stored in the .ipynb files.

Usage (from the project root):  python notebooks/execute_notebooks.py
"""
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent
for path in sorted(HERE.glob("0*.ipynb")):
    nb = nbformat.read(path, as_version=4)
    ExecutePreprocessor(timeout=1200, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(HERE)}})
    nbformat.write(nb, path)
    print("executed", path.name)
