"""LVIS (pronounced ‘el-vis’): is a new dataset for Large Vocabulary Instance Segmentation.
We collect over 2 million high-quality instance segmentation masks for over 1200 entry-level object categories in 164k images. LVIS API enables reading and interacting with annotation files,
visualizing annotations, and evaluating results.

"""
DOCLINES = (__doc__ or '')

import os.path
import sys
import pip

import setuptools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lvis"))

with open("requirements.txt") as f:
    reqs = f.read()

DISTNAME = "lvis"
DESCRIPTION = "Python API for LVIS dataset."
AUTHOR = "Agrim Gupta"
REQUIREMENTS = (reqs.strip().split("\n"),)


if __name__ == "__main__":
    setuptools.setup(
        name=DISTNAME,
        install_requires=REQUIREMENTS,
        packages=setuptools.find_packages(),
        version="0.5.3",
        description=DESCRIPTION,
        long_description=DOCLINES,
        long_description_content_type='text/markdown',
        author=AUTHOR
    )
