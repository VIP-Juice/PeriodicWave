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

COMMON_PACKAGES = [
    'absl-py==2.3.1',
    'attrs==25.4.0',
    'chex==0.1.91',
    'distrax==0.1.7',
    'folx @ git+https://github.com/microsoft/folx@d05c107028e3f88239ebf9e894d4a8c01abf90f6',
    'ipykernel==7.3.0',
    'matplotlib==3.11.0',
    'ml-collections==1.1.0',
    'optax==0.2.6',
    'pandas==2.3.3',
    'pyblock==0.6',
    'scipy==1.16.2',
    'typing_extensions==4.15.0',
]

CPU_PACKAGES = [
    'h5py==3.14.0',
    'jax==0.7.2',
    'jaxlib==0.7.2',
    'kfac-jax @ git+https://github.com/deepmind/kfac-jax@d9ecae99e588e4abbb0dd3d4e977d1266824e14c',
    'numpy==2.3.3',
    'tfp-nightly==0.26.0.dev20251007',
]

GPU_PACKAGES = [
    'h5py==3.15.1',
    'jax==0.8.0',
    'jax-cuda12-pjrt==0.8.0',
    'jax-cuda12-plugin[with-cuda]==0.8.0',
    'jaxlib==0.8.0',
    'kfac-jax @ git+https://github.com/deepmind/kfac-jax@55111a15e51b38a4cefbfd16cdda52472cda2632',
    'numpy==2.3.4',
    'tfp-nightly==0.26.0.dev20251017',
]

REQUIRED_PACKAGES = COMMON_PACKAGES


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
    extras_require={
        'cpu': CPU_PACKAGES,
        'gpu': GPU_PACKAGES,
        'testing': ['flake8', 'pylint', 'pytest', 'pytype'],
    },
    platforms=['any'],
    license='Apache 2.0',
)
