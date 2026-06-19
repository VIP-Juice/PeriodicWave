# Copyright 2020 DeepMind Technologies Limited.
# Modifications Copyright (c) 2025 Max Geier, Massachusetts Institute of Technology, MA, USA
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTICE: This file has been modified from the original DeepMind version.
# Changes:
# - Removed pyscf dependency because not needed for materials calculations

# ============================================================================
"""Setup for pip package."""

import unittest

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'absl-py',
    'attrs',
    'chex==0.1.91',
    'h5py==3.14.0',
    'folx @ git+https://github.com/microsoft/folx',
    'jax==0.7.2',
    'jaxlib==0.7.2',
    'kfac-jax @ git+https://github.com/deepmind/kfac-jax@d9ecae99e588e4abbb0dd3d4e977d1266824e14c',
    'ml-collections',
    'optax==0.2.6',
    'numpy==2.3.3',
    'pandas==2.3.3',
    'pyblock',
    'scipy==1.16.2',
    'typing_extensions',
    'distrax==0.1.7',
]


setup(
    name='periodicwave',
    version='0.0',
    description=(
        'Neural network variational Monte Carlo for solids'
    ),
    url='https://github.com/mg607/periodicwave',
    author='Max Geier, Khachatur Nazaryan',
    author_email='contact@deeppsi.ai',
    # Contained modules and scripts.
    entry_points={
        'console_scripts': [
            'periodicwave = periodicwave.main:main_wrapper',
        ],
    },
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    extras_require={'testing': ['flake8', 'pylint', 'pytest', 'pytype']},
    platforms=['any'],
    license='Apache 2.0',
)
