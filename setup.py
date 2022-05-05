#!/usr/bin/env python

"""Package setup file.
"""

from setuptools import setup, find_packages

setup(
    name='multirtd',
    version='1.0',
    description='Multi-robot Reachability-based Trajectory Planning',
    author='Adam Dai',
    author_email='adamdai97@gmail.com',
    url='https://github.com/adamdai/multirobot-planning',
    packages=find_packages(),
)