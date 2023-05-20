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

# PEP 440 won't accept the v in front, so here we remove it, strip the new line and decode the byte stream
VERSION_FROM_GIT_TAG = tags[-1][1:]

setup(
    name="condtionalconformal",
    version=VERSION_FROM_GIT_TAG,  # Required
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
