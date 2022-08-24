#!/usr/bin/env python

from setuptools import find_packages, setup

with open('VERSION') as f:
    version = f.read().strip()

setup(
    name='ood-metrics',
    version=version,
    author='Taylor Denouden',
    author_email='taylordenouden@gmail.com',
    packages=find_packages(),
    url='https://github.com/tayden/ood-metrics',
    license='MIT',
    description='Calculate common OOD detection metrics',
    long_description=open('./README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy>=1.15.3",
        "matplotlib>=3.0.1",
        "scikit-learn>=0.20.0",
    ],
)
