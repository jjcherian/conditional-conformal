#!/usr/bin/python
import subprocess
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
   long_description = fh.read()

# get all the git tags from the cmd line that follow our versioning pattern
git_tags = subprocess.Popen(
    ["git", "tag", "--list", "v*[0-9]", "--sort=version:refname"],
    stdout=subprocess.PIPE,
)
tags = git_tags.stdout.read()
git_tags.stdout.close()
tags = tags.decode("utf-8").split("\n")
tags.sort()


setup(
    name="condtionalconformal",
    version="0.0.4",  # Required
    setup_requires=["setuptools>=18.0"],
    packages=find_packages(exclude=["notebooks"]),  # Required
    install_requires=[
        "numpy",
        "scipy",
        "cvxpy",
	"scikit-learn",
	"matplotlib",
	"tqdm"
    ],
    description="This package enables conformal prediction with conditional guarantees.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jjcherian/conditional-conformal",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    author="John Cherian",
    author_email="jcherian@stanford.edu",
)
